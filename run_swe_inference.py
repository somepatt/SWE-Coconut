"""
Run inference on the SWE-bench Lite benchmark (300 tasks) and export predictions
in the JSON format accepted by the official evaluation server.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from loguru import logger

from src.config import load_config
from src.model import load_model_and_tokenizer
from src.utils import set_seed


SPECIAL_TOKENS = {
    "bot": "<bot>",
    "eot": "<eot>",
    "thought": "<thought>",
}

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "lite"

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
        help="Dataset split to evaluate (default: lite).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/swe_bench_predictions.json",
        help="Path to the JSON file that will store predictions.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
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
        default=None,
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
    return parser.parse_args()


def unwrap_generation_model(model) -> torch.nn.Module:
    """
    load_model_and_tokenizer may wrap CoconutModel with DistributedDataParallel.
    For inference we need the underlying Hugging Face AutoModel for `.generate`.
    """
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "model", model)


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
    return input_ids, attention_mask, len(sequence)


def generate_completion(
    model,
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
    model.eval()
    device = next(model.parameters()).device

    inputs, attention_mask, prefix_len = build_inputs(
        tokenizer,
        prompt,
        bot_id,
        eot_id,
        thought_id,
        num_thoughts,
        device,
    )

    use_sampling = num_beams == 1 and temperature > 0
    generation_kwargs = {
        "input_ids": inputs,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if use_sampling:
        generation_kwargs.update({"temperature": temperature, "do_sample": True, "top_p": 0.95})

    with torch.no_grad():
        sequences = model.generate(**generation_kwargs)

    completion_ids = sequences[0][prefix_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    completion_token_count = len(tokenizer.encode(completion, add_special_tokens=False))
    return completion, completion_token_count


def save_predictions(predictions: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    if args.num_thoughts is None:
        num_thoughts = config.training.continuous_thought_steps
    else:
        num_thoughts = args.num_thoughts

    logger.info(f"Loading dataset {args.dataset_name} ({args.dataset_split})...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    max_samples = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    if max_samples < len(dataset):
        logger.warning(f"Using only {max_samples} samples out of {len(dataset)}")
        dataset = dataset.select(range(max_samples))

    logger.info(f"Loaded {len(dataset)} tasks. Preparing model...")
    model_wrapper, tokenizer = load_model_and_tokenizer(config)
    model = unwrap_generation_model(model_wrapper)
    bot_id, eot_id, thought_id = get_special_token_ids(tokenizer)

    predictions: List[Dict[str, Any]] = []
    start_time = datetime.utcnow().isoformat()

    for idx, sample in enumerate(dataset):
        prompt = build_prompt(sample)
        completion, completion_token_len = generate_completion(
            model=model,
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

        predictions.append(
            {
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
        )

        if (idx + 1) % args.log_every == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} tasks")

    save_predictions(predictions, Path(args.output))

    # Clean up
    del model_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
