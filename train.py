import argparse
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.utils import setup_distributed

# Импортируем все кастомные модули из 'src'
try:
    from src.config import load_config, TrainingConfig
    from src.logger import setup_logger
    from src.utils import set_seed, get_device, setup_directories
    from src.model import load_model_and_tokenizer
    from src.data import get_dataset, get_cot_latent_dataset, MyCollator
    from src.optimizer import OptimizerManager
    from src.trainer import CoconutTrainer
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедись, что ты запускаешь скрипт из корня проекта и что твои"
          " модули находятся в директории 'src'.")
    exit(1)


def main(config_path: str):
    """
    Главная функция обучения.
    """
    
    # 1. Загрузка конфига и настройка окружения
    try:
        config: TrainingConfig = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Файл конфигурации не найден по пути: {config_path}")
        return
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфига: {e}")
        return

    device, rank, world_size = setup_distributed()
    config.data.batch_size //= world_size

    setup_logger(output_dir=config.output_dir, experiment_name=config.experiment_name)
    set_seed(config.seed + rank) # Разный seed для каждого процесса
    setup_directories(config) # Создает output_dir
    device = get_device()
    
    logger.info(f"Конфигурация успешно загружена из {config_path}")
    logger.info(f"Эксперимент: {config.experiment_name}")
    logger.info(f"Обучение будет запущено на устройстве: {device}")

    # 2. Загрузка модели и токенайзера
    # (Предполагается, что load_model_and_tokenizer в src/model.py
    # корректно добавляет токены '<bot>', '<eot>', '<thought>')
    logger.info("Загрузка модели и токенайзера...")
    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)
    
    # Получаем ID спец. токенов, необходимых для data.py
    try:
        latent_id = tokenizer.convert_tokens_to_ids('<thought>')
        start_id = tokenizer.convert_tokens_to_ids('<bot>')
        end_id = tokenizer.convert_tokens_to_ids('<eot>')
        
        # Проверяем, что все токены были добавлены и найдены
        if any(t is None or t == tokenizer.unk_token_id for t in [latent_id, start_id, end_id]):
            raise ValueError("Один или несколько спец. токенов (<bot>, <eot>, <thought>) не найдены.")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.error("Не удалось получить ID спец. токенов.")
        logger.error("Убедись, что 'load_model_and_tokenizer' в src/model.py "
                     "корректно добавляет ['<bot>', '<eot>', '<thought>'] "
                     "в токенайзер и что model.set_latent_token_id() "
                     "устанавливает ID для '={latent_id}")

    # 3. Загрузка базовых наборов данных
    logger.info("Загрузка базовых наборов данных...")
    try:
        # Пути должны быть указаны в твоем .yaml файле (например, 'config/default.yaml')
        # и соответствовать DataConfig в src/config.py
        base_train_dataset = get_dataset(
            dataset_name=config.data.dataset_name,
            split=config.data.split, # т.е. "train" из твоего DataConfig
            tokenizer=tokenizer
            )
        # base_val_dataset = get_dataset(config.data.val_path, tokenizer) # Для валидации
    except AttributeError as e:
        logger.error(f"Ошибка: Отсутствует 'dataset_name' или 'split' в конфиге: {e}. ")
        logger.error("Пожалуйста, добавь 'dataset_name' и 'split' "
                      "в секцию 'data' твоего YAML-конфига.")
        return
    except FileNotFoundError as e:
        logger.error(f"Файл данных не найден: {e}")
        return

    
        
    logger.info(f"Базовый трейн-сет загружен: {len(base_train_dataset)} сэмплов.")
    # logger.info(f"Базовый вал-сет загружен: {len(base_val_dataset)} сэмплов.")

    # 4. Создание Collator
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # 5. Создание Менеджера Оптимизатора и Тренера
    optimizer_manager = OptimizerManager(model, config)
    trainer = CoconutTrainer(model, tokenizer, config, optimizer_manager)

    # 6. Запуск многостадийного обучения
    logger.info("=" * 80)
    logger.info("Начало многостадийного обучения COCONUT...")
    logger.info(f"Всего стадий: {config.training.num_stages}")
    logger.info("=" * 80)
    
    for stage in range(config.training.num_stages + 1):
        
        # Создаем словарь конфигов, который ожидает data.py
        # Это мост между config.py и data.py
        data_processing_config = {
            "uniform_prob": config.training.uniform_prob,
            "max_latent_stage": config.training.num_stages,
            "pad_latent_to_max": config.training.pad_latent_to_max,
            "no_cot": False,
            "c_thought": config.training.continuous_thought_steps
        }
        
        logger.info(f"Подготовка данных для Стадии {stage}/{config.training.num_stages}...")
        
        # Создаем датасет для конкретной стадии
        train_dataset = get_cot_latent_dataset(
            scheduled_stage=stage,
            base_dataset=base_train_dataset,
            configs=data_processing_config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            no_special_marker=False,
            shuffle=False
        )

        sampler = DistributedSampler(
            base_train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True
        )
        
        # Создаем DataLoader для этой стадии
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            collate_fn=MyCollator(tokenizer, latent_id=latent_id),
            num_workers=config.data.num_workers,
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"DataLoader для стадии {stage} создан. "
                     f"Количество батчей: {len(train_loader)}")
        sampler.set_epoch(stage)
        
        # Запускаем одну стадию обучения
        trainer.train_stage(stage=stage, dataloader=train_loader)
        
        
        # (Опционально) Здесь можно добавить вызов валидации после каждой стадии
        # if base_val_dataset:
        #    logger.info(f"Запуск валидации после стадии {stage}...")
        #    # ... (логика валидации) ...

    logger.info("=" * 80)
    logger.info("Обучение COCONUT успешно завершено.")
    logger.info("=" * 80)
    
    # 7. Сохранение финальной модели
    # (Хотя CoconutTrainer сохраняет чекпоинты, финальное сохранение полезно)
    if rank == 0:
        try:
            final_save_dir = Path(config.output_dir) / "final_model"
            final_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем базовую модель (Peft или обычную)
            model.model.save_pretrained(str(final_save_dir))
            
            # Сохраняем токенайзер
            tokenizer.save_pretrained(str(final_save_dir))
            
            # Сохраняем конфиг, с которым модель обучалась
            config.to_yaml(str(final_save_dir / "config.yaml"))
            
            logger.info(f"Финальная модель сохранена в: {final_save_dir}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении финальной модели: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск обучения модели COCONUT")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Путь к YAML-файлу конфигурации обучения."
    )
    
    args = parser.parse_args()
    
    main(args.config)