# src/data.py
import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
from datasets import Dataset, load_dataset 
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from loguru import logger

# === НОВОЕ: тот же шаблон, что и в inference_ddp.py ===
PROMPT_TEMPLATE = """You are an autonomous software engineer tasked with fixing bugs.
Analyze the issue and produce a minimal unified diff that resolves the failure.
If you are unsure, still return your best attempt at a patch.

Repository: {repo}
Base commit: {base_commit}
Instance ID: {instance_id}

Problem statement:
{problem}

Provide ONLY the patch in a fenced diff block or raw unified diff.
"""

def build_train_prompt(sample: Dict) -> str:
    """Строим ровно тот же промпт, что и при инференсе."""
    repo = sample.get("repo") or sample.get("repo_name") or "unknown repo"
    base_commit = sample.get("base_commit") or sample.get("base_commit_id") or "unknown"
    problem = sample.get("problem_statement") or sample.get("prompt") or ""
    instance_id = sample.get("instance_id") or sample.get("id") or "NA"

    return PROMPT_TEMPLATE.format(
        repo=repo,
        base_commit=base_commit,
        instance_id=instance_id,
        problem=problem.strip(),
    ).strip() + "\n"


def get_dataset(
    dataset_name: str, 
    split: str, 
    tokenizer: PreTrainedTokenizerBase, 
    max_size: int = 20000,
    max_seq_length: int = 8192,  # фильтр по длине (должен совпадать с конфигом)
) -> Dataset:
    """Load and tokenize dataset from Hugging Face"""
    
    logger.info(f"Loading dataset '{dataset_name}' split '{split}' from Hugging Face...")

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Не удалось загрузить датасет '{dataset_name}': {e}")
        raise

    if max_size < len(dataset):
        logger.warning(f"Using a subset of {max_size} samples (full size: {len(dataset)})")
        dataset = dataset.select(range(max_size))

    def tokenize_sample(sample, idx):
        """
        Tokenizes a sample.
        """
        # === ВАЖНО: теперь используем тот же промпт, что в inference ===
        prompt = build_train_prompt(sample)

        # question_tokenized = tokenizer.encode(
        #     sample["problem_statement"] + "\n", add_special_tokens=True
        # )
        # Теперь:
        question_tokenized = tokenizer.encode(
            prompt,
            add_special_tokens=False  # как в inference_ddp.build_inputs
        )
        
        # 'patch' - это наши "шаги/ответ" (патч)
        patch_lines = sample.get("patch", "").split("\n")
        steps_tokenized = [
            tokenizer.encode(line + "\n", add_special_tokens=False)
            for line in patch_lines if line.strip()
        ]
        
        # Один EOS в конце как "конец патча"
        answer_tokenized = [tokenizer.eos_token_id]
        
        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": idx,
        }

    # 1. Токенизация
    dataset = dataset.map(
        tokenize_sample,
        with_indices=True,
        remove_columns=list(dataset.features),
        num_proc=32,
        desc="Tokenizing"
    )

    # 2. ✅ Фильтрация по длине (с учётом нового промпта)
    prev_len = len(dataset)
    
    def filter_long_samples(sample):
        # Общее количество токенов в патче (сумма длин шагов)
        patch_len = sum(len(step) for step in sample["steps_tokenized"])
        
        if patch_len > max_seq_length:
            return False
            
        total_len = len(sample["question_tokenized"]) + patch_len
        if total_len > max_seq_length:
             return False
             
        return True

    dataset = dataset.filter(
        filter_long_samples,
        num_proc=32,
        desc="Filtering by length"
    )
    
    logger.info(f"Filtered {prev_len - len(dataset)} samples exceeding {max_seq_length} tokens.")
    logger.info(f"Dataset loaded and filtered: {len(dataset)} samples")
    
    return dataset


@dataclass
class MyCollator:
    """Collate with KV cache optimization"""
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        if not features:
            return {}
            
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
        
        label_name = "label" if "label" in features[0].keys() else "labels"
        
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
            if label_name in features[0].keys()
            else None
        )
        
        if labels is not None and all(label is None for label in labels):
            labels = None
        
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
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
    
    # Берем макс длину из конфига или дефолт 8192
    max_seq_len = configs.get("max_seq_length", 8192)

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
        
        # Собираем токены
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

        # Собираем labels (маскируем вопрос и латентные токены через -100)
        labels = (
            [-100] * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens + n_additional_tokens
                + len(sample["question_tokenized"]):
            ]
        )

        # Обрезаем по длине
        tokens = tokens[:max_seq_len]
        labels = labels[:max_seq_len]

        return {
            "input_ids": tokens,
            "labels": labels,
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
