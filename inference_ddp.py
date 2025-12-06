"""
Run inference on the SWE-bench Lite benchmark (300 tasks) and export predictions
in the JSON format accepted by the official evaluation server.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

import torch.distributed as dist
from src.config import load_config
from src.model import load_model_and_tokenizer
from src.utils import set_seed, setup_distributed, get_world_size_and_rank


SPECIAL_TOKENS = {
    "bot": "<bot>",
    "eot": "<eot>",
    "thought": "<thought>",
}

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "dev"

PROMPT_TEMPLATE = """You are an autonomous software engineer tasked with fixing bugs.
Analyze the issue and produce a minimal unified diff that resolves the failure.
If you are unsure, still return your best attempt at a patch. You should give ONLY new patch

Repository: {repo}
Base commit: {base_commit}
Instance ID: {instance_id}

Problem statement:
{problem}

Provide ONLY the patch in a fenced diff block or raw unified diff.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench Lite inference with a trained COCONUT model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the training config describing the fine-tuned model.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset identifier on Hugging Face (default: SWE-bench Lite).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs_qwen2.5-7B-Instruct/swe_bench_predictions.json",
        help="Path to the JSON file that will store predictions.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of tasks to evaluate (Lite benchmark uses 300).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate for each task.",
    )
    parser.add_argument(
        "--num-thoughts",
        type=int,
        default=2,
        help="Override the number of <thought> tokens (defaults to config).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (ignored when num_beams > 1).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam size for generation. Use 1 to enable sampling.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress after this many tasks.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training.",
    )
    return parser.parse_args()


def get_special_token_ids(tokenizer) -> Tuple[int, int, int]:
    bot_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bot"])
    eot_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eot"])
    thought_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["thought"])

    missing = [name for name, idx in zip(SPECIAL_TOKENS.keys(), [bot_id, eot_id, thought_id]) if idx is None]
    if missing:
        raise ValueError(f"Missing special tokens required for inference: {missing}")
    return bot_id, eot_id, thought_id


def build_prompt(sample: Dict[str, Any]) -> str:
    repo = sample.get("repo") or sample.get("repo_name") or "unknown repo"
    base_commit = sample.get("base_commit") or sample.get("base_commit_id") or "unknown"
    problem = sample.get("problem_statement") or sample.get("prompt") or ""
    instance_id = sample.get("instance_id") or sample.get("id") or "NA"

    return PROMPT_TEMPLATE.format(
        repo=repo,
        base_commit=base_commit,
        problem=problem.strip(),
        instance_id=instance_id,
    ).strip() + "\n"


def build_inputs(
    tokenizer,
    prompt: str,
    bot_id: int,
    eot_id: int,
    thought_id: int,
    num_thoughts: int,
    device: torch.device,
):
    question_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    latent_tokens = [thought_id] * max(num_thoughts, 0)
    sequence = question_tokens + [bot_id] + latent_tokens + [eot_id]
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    position_ids = torch.arange(
        0, input_ids.size(1), dtype=torch.long, device=device
    ).unsqueeze(0)

    return input_ids, attention_mask, position_ids, len(sequence)


def coconut_generate(
    coconut_model: torch.nn.Module,
    tokenizer,
    prompt: str,
    bot_id: int,
    eot_id: int,
    thought_id: int,
    num_thoughts: int,
    max_new_tokens: int,
    temperature: float,
    num_beams: int,
) -> Tuple[str, int]:
    """
    Авторегрессивный Coconut-инференс:
    - строим префикс с <thought>-токенами;
    - на каждом шаге вызываем CoconutModel.forward;
    - берём последний логит, сэмплируем/аргмаксим токен;
    - дописываем его в конец и повторяем.
    """
    if num_beams != 1:
        raise ValueError("Coconut генератор сейчас поддерживает только num_beams=1")

    coconut_model.eval()
    device = next(coconut_model.parameters()).device

    input_ids, attention_mask, position_ids, prefix_len = build_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        bot_id=bot_id,
        eot_id=eot_id,
        thought_id=thought_id,
        num_thoughts=num_thoughts,
        device=device,
    )

    eos_id = tokenizer.eos_token_id
    use_sampling = temperature is not None and temperature > 0.0

    logger.info(f"Prefix length: {prefix_len}")
    logger.info(
        f"Generation kwargs (Coconut): max_new_tokens={max_new_tokens}, "
        f"temperature={temperature}, use_sampling={use_sampling}"
    )

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = coconut_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = outputs["logits"]
            last_logits = logits[:, -1, :]  # [batch=1, vocab]

            if use_sampling:
                # temperature scaling + top-p можно добавить, пока сделаем простое температурное сэмплирование
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)  # [1, 1]

            next_id = next_token.item()

            # Лог для отладки
            if step == 0:
                logger.info(f"First generated token id: {next_id} ({tokenizer.decode([next_id])!r})")

            # Если модель сразу предсказывает EOS — останавливаемся
            if next_id == eos_id:
                break

            # Дописываем токен к последовательности
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Маска и позиции обновляются по длине
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            position_ids = torch.arange(
                0, input_ids.size(1), dtype=torch.long, device=device
            ).unsqueeze(0)

    # Отделяем completion от префикса
    completion_ids = input_ids[0][prefix_len:]
    logger.info(f"Generated IDs tail: {completion_ids.tolist()}")

    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True
    ).strip()

    # Считаем длину в токенах для репорта
    completion_token_count = len(
        tokenizer.encode(completion, add_special_tokens=False)
    )

    return completion, completion_token_count


def generate_completion(
    model,          # это CoconutModel или DDP(CoconutModel)
    tokenizer,
    prompt: str,
    bot_id: int,
    eot_id: int,
    thought_id: int,
    num_thoughts: int,
    max_new_tokens: int,
    temperature: float,
    num_beams: int,
) -> Tuple[str, int]:
    return coconut_generate(
        coconut_model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        bot_id=bot_id,
        eot_id=eot_id,
        thought_id=thought_id,
        num_thoughts=num_thoughts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_beams=num_beams,
    )


def save_predictions(predictions: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    minimal_predictions = [
        {
            "instance_id": pred["instance_id"],
            "model_patch": pred["model_patch"],
            "model_name_or_path": pred["model_name_or_path"]
        }
        for pred in predictions
    ]
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(minimal_predictions, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    args = parse_args()
    setup_distributed()
    world_size, rank = get_world_size_and_rank()
    
    config = load_config(args.config)
    set_seed(config.seed)

    if args.num_thoughts is None:
        num_thoughts = config.training.continuous_thought_steps
    else:
        num_thoughts = args.num_thoughts

    logger.info(f"Loading dataset {args.dataset_name} ({args.dataset_split})...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    logger.info(f"[Rank {rank}] Loaded {len(dataset)} tasks.")

    max_samples = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    if max_samples < len(dataset):
        logger.warning(f"Using only {max_samples} samples out of {len(dataset)}")
        dataset = dataset.select(range(max_samples))

    logger.info(f"Loaded {len(dataset)} tasks. Preparing model...")
    model_wrapper, tokenizer = load_model_and_tokenizer(config)
    coconut_model = model_wrapper
    bot_id, eot_id, thought_id = get_special_token_ids(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions: List[Dict[str, Any]] = []
    start_time = datetime.utcnow().isoformat()

    for idx, sample in tqdm(enumerate(dataset)):
        prompt = build_prompt(sample)
        
        completion, completion_token_len = generate_completion(
            model=coconut_model,
            tokenizer=tokenizer,
            prompt=prompt,
            bot_id=bot_id,
            eot_id=eot_id,
            thought_id=thought_id,
            num_thoughts=num_thoughts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
        )

        instance_id = sample.get("instance_id") or sample.get("id") or f"sample-{idx}"

        prediction = {
            "instance_id": instance_id,
            "model_patch": completion,
            "model_patch_token_len": completion_token_len,
            "model_name_or_path": config.model.name,
            "model_checkpoint": config.model.resume_from_checkpoint,
            "generation_parameters": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "num_beams": args.num_beams,
                "num_thought_tokens": num_thoughts,
            },
            "prompt_used": prompt,
            "generated_at": start_time,
        }

        predictions.append(prediction)

        if idx < 2 or (idx + 1) % args.log_every == 0:
            logger.info(f"[Rank {rank}] Task {idx + 1}/{len(dataset)}")
            logger.info(f"[Rank {rank}] Instance ID: {instance_id}")
            logger.info(f"[Rank {rank}] Patch preview:\n{completion[:400]}\n{'-'*40}")

    gathered_predictions = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_predictions, predictions)

    if rank == 0:
        all_preds = [p for sublist in gathered_predictions for p in sublist]
        save_predictions(all_preds, Path(args.output))

    # Clean up
    del coconut_model
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

