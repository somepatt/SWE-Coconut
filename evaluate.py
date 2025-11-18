"""
Evaluation script for the COCONUT project.

The script compares the raw base model (before LoRA training) and the
fine-tuned model that loads LoRA adapters from the checkpoint path specified in
the config. A prompt is fed to both models and their generations are logged.
"""

import argparse
from typing import Tuple

import torch
from loguru import logger

from src.config import load_config, TrainingConfig
from src.model import load_model_and_tokenizer


SPECIAL_TOKENS = {
    "bot": "<bot>",
    "eot": "<eot>",
    "thought": "<thought>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COCONUT models before and after training.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the training/evaluation config.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt that will be sent to the models.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--num-thoughts",
        type=int,
        default=None,
        help="Override the number of <thought> tokens (defaults to config.training.continuous_thought_steps).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (only used when num_beams=1).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search. Set to 1 to disable beam search.",
    )
    return parser.parse_args()


def unwrap_generation_model(model) -> torch.nn.Module:
    """
    load_model_and_tokenizer returns CoconutModel or DDP-wrapped CoconutModel.
    We unwrap it to get to the underlying HuggingFace AutoModel that implements
    `.generate`.
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
        raise ValueError(f"Missing special tokens for evaluation: {missing}")
    return bot_id, eot_id, thought_id


def build_inputs(tokenizer, prompt: str, bot_id: int, eot_id: int, thought_id: int, num_thoughts: int, device: torch.device):
    question_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    latent_tokens = [thought_id] * max(num_thoughts, 0)
    full_ids = question_tokens + [bot_id] + latent_tokens + [eot_id]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask, len(full_ids)


def generate_response(
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
) -> str:
    model.eval()
    device = next(model.parameters()).device

    inputs, attention_mask, input_length = build_inputs(
        tokenizer,
        prompt,
        bot_id,
        eot_id,
        thought_id,
        num_thoughts,
        device,
    )

    use_sampling = num_beams == 1 and temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature if use_sampling else None,
            do_sample=use_sampling,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_ids = output_ids[0][input_length:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion.strip()


def evaluate_single_model(
    label: str,
    config: TrainingConfig,
    prompt: str,
    num_thoughts: int,
    max_new_tokens: int,
    temperature: float,
    num_beams: int,
) -> str:
    logger.info(f"Loading {label}...")
    model_wrapper, tokenizer = load_model_and_tokenizer(config)
    model = unwrap_generation_model(model_wrapper)

    bot_id, eot_id, thought_id = get_special_token_ids(tokenizer)
    response = generate_response(
        model=model,
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

    logger.info(f"[{label}] response:\n{response}")

    # Free the model quickly so the next evaluation fits in memory.
    del model_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


def main():
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    if args.num_thoughts is None:
        num_thoughts = config.training.continuous_thought_steps
    else:
        num_thoughts = args.num_thoughts

    if num_thoughts < 0:
        raise ValueError("Number of thoughts must be non-negative.")

    # Baseline: disable LoRA weights to emulate pre-training state.
    base_config = config.model_copy(deep=True)
    base_config.model.use_lora = False
    base_config.model.resume_from_checkpoint = None

    baseline_output = evaluate_single_model(
        label="BASELINE (no training)",
        config=base_config,
        prompt=args.prompt,
        num_thoughts=num_thoughts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )

    checkpoint_path = config.model.resume_from_checkpoint
    if not checkpoint_path:
        raise ValueError(
            "No checkpoint path found in config.model.resume_from_checkpoint. "
            "Cannot evaluate the trained model."
        )

    trained_config = config.model_copy(deep=True)
    trained_output = evaluate_single_model(
        label=f"FINE-TUNED ({checkpoint_path})",
        config=trained_config,
        prompt=args.prompt,
        num_thoughts=num_thoughts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )

    logger.info("=" * 80)
    logger.info("Evaluation complete. Prompt and responses:")
    logger.info(f"Prompt:\n{args.prompt}")
    logger.info(f"Baseline output:\n{baseline_output}")
    logger.info(f"Fine-tuned output:\n{trained_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
