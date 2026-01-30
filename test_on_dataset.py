"""
Interactive Dataset Testing for Coconut Models
===============================================

Loads the training dataset (trajectory files or HuggingFace), displays
available samples, and lets you run Coconut inference on any sample to
compare the model output with the ground-truth patch.

USAGE:
    # Basic — uses the config to locate the dataset and model
    python test_on_dataset.py \\
        --config config/simcot_qwen3_8b_a100.yaml \\
        --model path/to/merged_model

    # With LoRA adapter (not merged)
    python test_on_dataset.py \\
        --config config/simcot_qwen3_8b_a100.yaml \\
        --model Qwen/Qwen3-0.6B \\
        --adapter outputs_simcot_qwen3_8b/stage_3/step_4000

    # Override the number of thoughts or dataset
    python test_on_dataset.py \\
        --config config/simcot_qwen3_8b_a100.yaml \\
        --model path/to/merged_model \\
        --num-thoughts 12 \\
        --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
#  Dataset loading (standalone, no dependency on src.data for raw display)
# ---------------------------------------------------------------------------

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


def _split_assistant_turn(content: str) -> Optional[Dict[str, str]]:
    """Extract thought and code block from an assistant message."""
    match = re.search(r"```[^\n]*\n(.*?)```", content, flags=re.DOTALL)
    if not match:
        return None
    thought_text = content[: match.start()].strip()
    if thought_text.startswith("THOUGHT:"):
        thought_text = thought_text[len("THOUGHT:") :].strip()
    code_text = match.group(1).strip()
    if not code_text:
        return None
    return {"thought": thought_text, "code": code_text}


def _looks_like_observation(content: str) -> bool:
    stripped = content.lstrip()
    return stripped.startswith(("<returncode>", "<output>", "<warning>", "<error>"))


def _format_chat_messages(messages: List[Dict]) -> str:
    parts: List[str] = []
    for message in messages:
        role = message.get("role") or "user"
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and _looks_like_observation(content):
            label = "Observation"
        else:
            label = role.capitalize()
        parts.append(f"{label}:\n{content}")
    return "\n\n".join(parts).strip()


def load_trajectory_samples(dataset_path: str, max_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Load trajectory ``.traj.json`` files and extract (prompt, patch, thoughts) tuples.

    Each trajectory file may produce multiple samples (one per assistant step
    with incremental context accumulation).
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    files = sorted(path.glob("*.traj.json"))
    if not files:
        raise FileNotFoundError(f"No .traj.json files in: {dataset_path}")

    samples: List[Dict[str, Any]] = []

    for file in files:
        if len(samples) >= max_samples:
            break
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue

        messages = data.get("messages", [])
        instance_id = data.get("instance_id", file.stem)

        accumulated_context: List[Dict] = []
        step_counter = 0

        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content") or ""
                split = _split_assistant_turn(content)

                if split and split["code"]:
                    prompt = _format_chat_messages(accumulated_context)
                    if prompt:
                        samples.append(
                            {
                                "instance_id": f"{instance_id}-step{step_counter}",
                                "prompt": prompt,
                                "patch": split["code"],
                                "thought": split["thought"] or "",
                                "source_file": file.name,
                            }
                        )
                        step_counter += 1

                    if len(samples) >= max_samples:
                        break

            accumulated_context.append(message)

    print(f"Loaded {len(samples)} samples from {len(files)} trajectory files.")
    return samples


def load_hf_samples(dataset_name: str, split: str, max_samples: int = 200) -> List[Dict[str, Any]]:
    """Load samples from a HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    samples: List[Dict[str, Any]] = []

    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        repo = row.get("repo") or row.get("repo_name") or "unknown"
        base_commit = row.get("base_commit") or "unknown"
        problem = row.get("problem_statement") or row.get("prompt") or ""
        instance_id = row.get("instance_id") or f"sample-{i}"
        patch = row.get("patch") or ""

        prompt = PROMPT_TEMPLATE.format(
            repo=repo,
            base_commit=base_commit,
            instance_id=instance_id,
            problem=problem.strip(),
        ).strip() + "\n"

        samples.append(
            {
                "instance_id": instance_id,
                "prompt": prompt,
                "patch": patch,
                "thought": "",
                "source_file": dataset_name,
            }
        )

    print(f"Loaded {len(samples)} samples from HuggingFace dataset '{dataset_name}'.")
    return samples


def load_samples(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
    """Auto-detect dataset type (local trajectories vs HuggingFace) and load."""
    if Path(dataset_name).exists():
        return load_trajectory_samples(dataset_name, max_samples)
    return load_hf_samples(dataset_name, split, max_samples)


# ---------------------------------------------------------------------------
#  Display helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f" ... [{len(text)} chars total]"


def display_sample_list(samples: List[Dict[str, Any]], page_size: int = 20, offset: int = 0):
    """Print a paginated table of samples."""
    end = min(offset + page_size, len(samples))
    print(f"\n{'='*90}")
    print(f" SAMPLES  ({offset+1}–{end} of {len(samples)})")
    print(f"{'='*90}")
    print(f" {'#':>4}  {'Instance ID':<40}  {'Prompt (preview)'}")
    print(f" {'-'*4}  {'-'*40}  {'-'*40}")

    for i in range(offset, end):
        s = samples[i]
        iid = truncate(s["instance_id"], 38)
        preview = truncate(s["prompt"].replace("\n", " "), 40)
        print(f" {i:>4}  {iid:<40}  {preview}")

    print(f"{'='*90}")
    if end < len(samples):
        print(f" (Enter 'n' for next page, or type a sample number)")


def display_sample_detail(sample: Dict[str, Any]):
    """Print full detail of a single sample."""
    print(f"\n{'='*90}")
    print(f" SAMPLE: {sample['instance_id']}")
    print(f" Source: {sample.get('source_file', 'N/A')}")
    print(f"{'='*90}")

    print(f"\n--- PROMPT ({len(sample['prompt'])} chars) ---")
    if len(sample["prompt"]) > 2000:
        print(sample["prompt"][:2000])
        print(f"... [truncated, {len(sample['prompt'])} chars total]")
    else:
        print(sample["prompt"])

    if sample.get("thought"):
        print(f"\n--- THOUGHT (ground truth) ---")
        print(truncate(sample["thought"], 500))

    print(f"\n--- GROUND TRUTH PATCH ---")
    if len(sample["patch"]) > 3000:
        print(sample["patch"][:3000])
        print(f"... [truncated, {len(sample['patch'])} chars total]")
    else:
        print(sample["patch"])


def display_inference_result(result: Dict[str, Any], elapsed: float):
    """Print the model's generated output."""
    print(f"\n--- MODEL OUTPUT ---")
    print(result["text"])
    print(f"\n--- STATS ---")
    print(
        f"  tokens:        {result['tokens']}\n"
        f"  finish_reason: {result['finish_reason']}\n"
        f"  time:          {elapsed:.2f}s\n"
        f"  tok/s:         {result['tokens'] / max(elapsed, 1e-6):.1f}"
    )


# ---------------------------------------------------------------------------
#  Interactive loop
# ---------------------------------------------------------------------------

def interactive_loop(engine, samples: List[Dict[str, Any]], args):
    """
    Main interactive session.

    Commands:
        <number>  — select sample by index
        n         — next page
        p         — previous page
        q / exit  — quit
        r         — re-display current sample
    """
    page_size = 20
    offset = 0
    current_sample = None

    display_sample_list(samples, page_size, offset)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break

        if user_input.lower() == "n":
            offset = min(offset + page_size, len(samples) - page_size)
            offset = max(offset, 0)
            display_sample_list(samples, page_size, offset)
            continue

        if user_input.lower() == "p":
            offset = max(offset - page_size, 0)
            display_sample_list(samples, page_size, offset)
            continue

        if user_input.lower() == "r" and current_sample is not None:
            display_sample_detail(current_sample)
            continue

        if user_input.lower() == "list":
            display_sample_list(samples, page_size, offset)
            continue

        # Try to parse as sample index
        try:
            idx = int(user_input)
        except ValueError:
            print("Unknown command. Enter a sample number, 'n', 'p', 'list', or 'q'.")
            continue

        if idx < 0 or idx >= len(samples):
            print(f"Index out of range. Valid range: 0–{len(samples) - 1}")
            continue

        current_sample = samples[idx]
        display_sample_detail(current_sample)

        # Ask whether to run inference
        try:
            run = input("\nRun inference on this sample? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if run in ("", "y", "yes"):
            print("\nRunning Coconut inference ...")
            t0 = time.time()
            result = engine.generate(
                prompt=current_sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            elapsed = time.time() - t0
            display_inference_result(result, elapsed)


# ---------------------------------------------------------------------------
#  Non-interactive batch mode
# ---------------------------------------------------------------------------

def batch_mode(engine, samples: List[Dict[str, Any]], indices: List[int], args):
    """Run inference on specific sample indices and print results."""
    for idx in indices:
        if idx < 0 or idx >= len(samples):
            print(f"[WARN] Index {idx} out of range, skipping.")
            continue

        sample = samples[idx]
        display_sample_detail(sample)

        print(f"\nRunning Coconut inference ...")
        t0 = time.time()
        result = engine.generate(
            prompt=sample["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        elapsed = time.time() - t0
        display_inference_result(result, elapsed)
        print("\n" + "=" * 90 + "\n")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive testing of Coconut models on the training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config (provides dataset path, num_stages, c_thought, etc.)
    p.add_argument(
        "--config",
        type=str,
        default="config/simcot_qwen3_8b_a100.yaml",
        help="Path to the training config YAML.",
    )

    # Model (overrides config if provided)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path or HF hub ID of the model. If omitted, taken from config.",
    )
    p.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path.")
    p.add_argument("--num-thoughts", type=int, default=None, help="Override number of thoughts.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda")

    # Dataset overrides
    p.add_argument("--dataset", type=str, default=None, help="Override dataset path/name.")
    p.add_argument("--split", type=str, default=None, help="Override dataset split.")
    p.add_argument("--max-samples", type=int, default=100, help="Max samples to load.")

    # Generation
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)

    # Non-interactive mode
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated sample indices for batch (non-interactive) mode.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load config ----
    from src.config import load_config

    config = load_config(args.config)

    # ---- Resolve parameters ----
    model_path = args.model or config.model.name
    dataset_name = args.dataset or config.data.dataset_name
    dataset_split = args.split or config.data.split

    if args.num_thoughts is not None:
        num_thoughts = args.num_thoughts
    else:
        # Compute from config: num_stages * continuous_thought_steps
        num_thoughts = config.training.num_stages * config.training.continuous_thought_steps

    print(f"Model:       {model_path}")
    print(f"Adapter:     {args.adapter or 'None (using merged model)'}")
    print(f"Dataset:     {dataset_name}")
    print(f"Num thoughts: {num_thoughts}")
    print()

    # ---- Load dataset ----
    samples = load_samples(dataset_name, dataset_split, args.max_samples)
    if not samples:
        print("No samples loaded. Check the dataset path.")
        sys.exit(1)

    # ---- Load Coconut engine ----
    from serve_coconut import CoconutEngine

    engine = CoconutEngine(
        model_path=model_path,
        adapter_path=args.adapter,
        num_thoughts=num_thoughts,
        device=args.device,
        dtype=args.dtype,
    )

    # ---- Run ----
    if args.indices is not None:
        indices = [int(x.strip()) for x in args.indices.split(",")]
        batch_mode(engine, samples, indices, args)
    else:
        interactive_loop(engine, samples, args)


if __name__ == "__main__":
    main()
