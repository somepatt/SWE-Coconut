import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from loguru import logger

Понял. Ты хочешь использовать dataset_name: "SWE-bench/SWE-smith" из твоего config.py, но текущий код в train.py и data.py несовместим с этим.

train.py ожидает локальный путь (config.data.train_path), а data.py пытается открыть этот путь как JSON-файл (json.load(open(path))).

Я исправлю data.py и train.py, чтобы они корректно загружали и обрабатывали датасет "SWE-bench/SWE-smith" с Hugging Face, используя твой DataConfig.

1. data.py (Исправлено)
Я полностью переписал функцию get_dataset, чтобы она:

Импортировала load_dataset из библиотеки datasets.

Использовала dataset_name и split из конфига.

Исправил tokenize_sample, так как в "SWE-bench/SWE-smith" нет поля "answer". Мы будем использовать "problem_statement" как вопрос, а "patch" — как "мысли" (CoT) и ответ.

Python

# src/data.py
import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
import torch.distributed as dist
# ✅ ИМПОРТ
from datasets import Dataset, load_dataset 
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from loguru import logger

# ✅ ФУНКЦИЯ ПЕРЕПИСАНА
def get_dataset(
    dataset_name: str, 
    split: str, 
    tokenizer: PreTrainedTokenizerBase, 
    max_size=1000000000
) -> Dataset:
    """Load and tokenize dataset from Hugging Face"""
    
    logger.info(f"Loading dataset '{dataset_name}' split '{split}' from Hugging Face...")

    try:
        # Загружаем датасет с Hugging Face
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Не удалось загрузить датасет '{dataset_name}': {e}")
        logger.error("Убедись, что у тебя есть доступ к интернету и 'datasets' "
                     "библиотека установлена (pip install datasets).")
        raise

    if max_size < len(dataset):
        logger.warning(f"Using a subset of {max_size} samples (full size: {len(dataset)})")
        dataset = dataset.select(range(max_size))

    def tokenize_sample(sample, idx):
        """
        Tokenizes a sample from 'SWE-bench/SWE-smith'
        Question = problem_statement
        Steps (CoT) = lines from 'patch'
        Answer = [EOS] (since 'patch' is the full output)
        """
        
        # 'problem_statement' - это наш вопрос
        question_tokenized = tokenizer.encode(
            sample["problem_statement"] + "\n", add_special_tokens=True
        )
        
        # 'patch' - это наши "мысли" (Chain-of-Thought), разбитые по строкам
        patch_lines = sample.get("patch", "").split("\n")
        steps_tokenized = [
            tokenizer.encode(line + "\n", add_special_tokens=False)
            for line in patch_lines if line.strip()
        ]
        
        answer_tokenized = [tokenizer.eos_token_id]
        
        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": idx, # Добавляем 'idx'
        }

    # Применяем токенизацию
    dataset = dataset.map(
        tokenize_sample,
        with_indices=True, # Передаем 'idx' в функцию
        remove_columns=list(dataset.features), # Удаляем старые колонки
        num_proc=32
    )

    logger.info(f"Dataset loaded and tokenized: {len(dataset)} samples")
    return dataset


@dataclass
class MyCollator:
    """Collate with KV cache optimization"""


    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"
        
        # Find earliest latent position to pad for KV cache reuse
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]
        
        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                
                if "labels" in feature:
                    feature["labels"] = [
                        self.label_pad_token_id
                    ] * n_tok_pad + feature["labels"]
                
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]
        
        return_tensors = "pt"
        label_name = "label" if "label" in features.keys() else "labels"
        
        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]
        
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            return_tensors=return_tensors,
        )
        
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features.keys()
            else None
        )
        
        if labels is not None and all(label is None for label in labels):
            labels = None
        
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features.keys()
            else None
        )
        
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = torch.tensor([
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ], dtype=torch.int64)
        
        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = torch.tensor(
                [
                    [0] * (max_pos_length - len(position_id)) + position_id
                    for position_id in position_ids
                ], 
                dtype=torch.int64
            )
        
        return batch
def get_question_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    ):
    """Dataset for generation (evaluation)"""


    def process_dataset(sample):
        if configs.get("pad_latent_to_max", False):
            max_latent_stage = configs.get("max_latent_stage", 5)
        else:
            max_latent_stage = min(
                configs.get("max_latent_stage", 5),
                len(sample["steps_tokenized"])
            )
        
        k = min(max_latent_stage, scheduled_stage)
        k *= configs.get("c_thought", 1)
        
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )
        
        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask":  [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset.map(
        process_dataset,
        remove_columns=list(base_dataset.features),
        num_proc=32
    )
def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
    ):
    """Dataset for training with latent tokens"""

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        if random.random() < configs.get("uniform_prob", 0.0):
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage
        
        if scheduled_stage_to_train > configs.get("max_latent_stage", 5):
            n_skip_steps = 10000
            if configs.get("pad_latent_to_max", False):
                n_latent_tokens = configs.get("max_latent_stage", 5)
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]),
                    configs.get("max_latent_stage", 5)
                )
        else:
            n_skip_steps = scheduled_stage_to_train
            n_latent_tokens = scheduled_stage_to_train
        
        if configs.get("no_cot", False):
            n_skip_steps = 100
            n_latent_tokens = 0
        
        n_latent_tokens *= configs.get("c_thought", 1)
        
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(itertools.chain.from_iterable(
                sample["steps_tokenized"][n_skip_steps:]
            ))
            + sample["answer_tokenized"]
        )
        
        return {
            "input_ids": tokens,
            "labels": (
                [-100] * (
                    len(sample["question_tokenized"])
                    + n_latent_tokens
                    + n_additional_tokens
                )
                + tokens[
                    n_latent_tokens + n_additional_tokens
                    + len(sample["question_tokenized"]):
                ]
            ),
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    processed_dataset = base_dataset.map(
        process_dataset,
        remove_columns=list(base_dataset.features),
        num_proc=32
    )

    if shuffle:
        processed_dataset = processed_dataset.shuffle()

    return processed_dataset