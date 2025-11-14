"""
Evaluation script for COCONUT model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) # Убедись, что путь до 'src' верный

import torch
# 🔻 ИМПОРТИРУЕМ НУЖНЫЕ КЛАССЫ
from src.model import load_model_and_tokenizer
from src.config import TrainingConfig
from loguru import logger

def generate_solution(
    model,
    tokenizer,
    problem: str,
    bot_token_id: int,    # ID <bot>
    eot_token_id: int,    # ID <eot>
    thought_token_id: int, # ID <thought>
    num_thoughts: int = 10, # Количество "мыслей"
    max_length: int = 512,
):
    """Generate solution for given problem using COCONUT"""
    
    # ✅ СОЗДАЕМ ПРОМПТ С ЛАТЕНТНЫМИ ТОКЕНАМИ
    # Это реализует инференс-процесс из статьи [cite: 146, 148]
    prompt_text = f"Fix this bug:\n{problem}\n\nSolution:"
    
    question_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Формируем вход для COCONUT
    input_ids = (
        question_tokens 
        + [bot_token_id] 
        + [thought_token_id] * num_thoughts 
        + [eot_token_id]
    )
    
    inputs = torch.tensor([input_ids]).to("cuda")
    attention_mask = torch.ones_like(inputs) # Маска на все токены
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_length, # Используй max_new_tokens
            num_beams=3,
            early_stopping=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id # Важно для generate
        )
    
    # Декодируем, пропуская *весь* инпут-промпт
    solution = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    return solution

def main():
    model_dir = "./outputs/final_model"
    
    try:
        config = TrainingConfig.from_yaml(f"{model_dir}/config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml не найден. Загрузка невозможна.")
        return
        
    # Говорим, что базовая модель теперь лежит в model_dir
    config.model.name = model_dir 
    
    # Загружаем модель с правильной оберткой
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Получаем ID спец. токенов
    bot_id = tokenizer.convert_tokens_to_ids("<bot>")
    eot_id = tokenizer.convert_tokens_to_ids("<eot>")
    thought_id = tokenizer.convert_tokens_to_ids("<thought>")
    
    model = model.to("cuda")
    model.eval()
    
    # Test
    problem = "The function returns None instead of an empty list"
    
    solution = generate_solution(
        model, 
        tokenizer, 
        problem,
        bot_id,
        eot_id,
        thought_id,
        num_thoughts=config.training.continuous_thought_steps # Берем из конфига
    )
    
    logger.info(f"Problem: {problem}")
    logger.info(f"Solution: {solution}")

if __name__ == "__main__":
    main()