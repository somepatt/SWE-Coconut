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
    max_seq_length: int = 8192,
) -> Dataset:
    """
    Загружаем SWE-smith и превращаем его в:
      - question_tokenized: prompt (репо + проблема)
      - steps_tokenized: ПУСТО (нет явного CoT)
      - answer_tokenized: ПОЛНЫЙ patch (diff) + EOS

    Модель будет учиться предсказывать именно patch.
    """
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
        ВАЖНО: CoT нет, поэтому:
          - steps_tokenized = []
          - весь patch идёт в answer_tokenized
        """
        prompt = build_train_prompt(sample)

        question_tokenized = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        patch_text = sample.get("patch", "")

        answer_tokenized = tokenizer.encode(
            patch_text + tokenizer.eos_token,
            add_special_tokens=False,
        )

        steps_tokenized = []

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": idx,
        }

    dataset = dataset.map(
        tokenize_sample,
        with_indices=True,
        remove_columns=list(dataset.features),
        num_proc=32,
        desc="Tokenizing",
    )

    prev_len = len(dataset)

    def filter_long_samples(sample):
        patch_len = len(sample["answer_tokenized"])
        total_len = len(sample["question_tokenized"]) + patch_len

        if patch_len == 0:
            return False

        if total_len > max_seq_length:
            return False

        return True

    dataset = dataset.filter(
        filter_long_samples,
        num_proc=32,
        desc="Filtering by length",
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
            batch["labels"] = torch.tensor(
                [
                    label + [self.label_pad_token_id] * (max_label_length - len(label))
                    for label in labels
                ],
                dtype=torch.int64,
            )

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = torch.tensor(
                [
                    [0] * (max_pos_length - len(position_id)) + position_id
                    for position_id in position_ids
                ],
                dtype=torch.int64,
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker: bool = False,
):
    """
    Dataset для генерации (evaluation).

    Так как CoT нет, количество latent-токенов определяется только:
      - current stage (scheduled_stage)
      - max_latent_stage
      - c_thought

    k = min(stage, max_latent_stage) * c_thought
    """

    max_latent_stage = configs.get("max_latent_stage", 5)
    c_thought = configs.get("c_thought", 1)

    def process_dataset(sample):
        k = min(scheduled_stage, max_latent_stage) * c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset.map(
        process_dataset,
        remove_columns=list(base_dataset.features),
        num_proc=32,
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker: bool = False,
    shuffle: bool = False,
):
    """
    Dataset для обучения с latent-токенами, БЕЗ явного CoT.

    Что происходит:
      - Вход: question + <bot> + <thought>*N + <eot> + patch + eos
      - Labels: mask(prompt + bot/thought/eot) + patch+eos

    Количество latent-токенов N растёт с каждым stage:
      N = min(stage, max_latent_stage) * c_thought
    """

    n_additional_tokens = 0 if no_special_marker else 2

    max_seq_len = configs.get("max_seq_length", 8192)
    max_latent_stage = configs.get("max_latent_stage", 5)
    c_thought = configs.get("c_thought", 1)
    no_cot_flag = configs.get("no_cot", False)

    def process_dataset(sample):
        if no_cot_flag:
            n_latent_tokens = 0
        else:
            stage_clamped = min(scheduled_stage, max_latent_stage)
            n_latent_tokens = stage_clamped * c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + sample["answer_tokenized"]
        )

        prefix_len = (
            len(sample["question_tokenized"])
            + n_latent_tokens
            + n_additional_tokens
        )
        labels = [-100] * prefix_len + tokens[prefix_len:]

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
        num_proc=32,
    )

    if shuffle:
        processed_dataset = processed_dataset.shuffle()

    return processed_dataset
