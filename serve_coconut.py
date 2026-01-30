"""
Coconut Inference Server
========================

Drop-in replacement for ``vllm serve`` for models trained with the
Coconut (Chain of Continuous Thought) paradigm.

WHY NOT VANILLA vLLM?
    The Coconut inference process requires an *iterative* forward pass where
    each <thought> token's embedding is replaced with the hidden state from
    the previous position (see Hao et al., 2024 — "Training Large Language
    Models to Reason in a Continuous Latent Space").  Standard vLLM performs
    a single-pass prefill and does not support modifying embeddings between
    positions, so a custom engine is necessary.

USAGE:
    # Serve with OpenAI-compatible API (like ``vllm serve``):
    python serve_coconut.py \\
        --model path/to/merged_model \\
        --num-thoughts 12 \\
        --port 8000

    # Or use the engine directly from Python:
    from serve_coconut import CoconutEngine
    engine = CoconutEngine("path/to/model", num_thoughts=12)
    result = engine.generate("Fix the bug in ...", max_new_tokens=512)
    print(result["text"])

API ENDPOINTS (identical to vLLM):
    POST /v1/completions
    POST /v1/chat/completions
    GET  /v1/models
    GET  /health
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional dependencies — server mode requires FastAPI + Uvicorn
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
    import uvicorn

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

# Optional — LoRA adapter merging
try:
    from peft import PeftModel

    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"
THOUGHT_TOKEN = "<thought>"


# ---------------------------------------------------------------------------
#  Coconut Inference Engine
# ---------------------------------------------------------------------------

class CoconutEngine:
    """
    Core inference engine for Coconut (Chain of Continuous Thought) models.

    Implements the iterative forward pass described in Figure 1 of the paper:
    for each ``<thought>`` token, the model runs a forward pass and uses the
    last hidden state of the **previous** position as the input embedding for
    the current thought.  After all thoughts are processed, standard
    autoregressive decoding is used to generate the answer.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        num_thoughts: int = 12,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_seq_len: int = 8192,
    ):
        """
        Args:
            model_path:        Path or HF hub ID of the base / merged model.
            adapter_path:      Optional path to LoRA adapter dir (merged on load).
            num_thoughts:      Number of ``<thought>`` tokens to insert.
            device:            ``"cuda"``, ``"cpu"``, or ``"auto"``.
            dtype:             ``"bfloat16"`` | ``"float16"`` | ``"float32"``.
            trust_remote_code: Passed to ``from_pretrained``.
            max_seq_len:       Maximum total sequence length.
        """
        self.num_thoughts = num_thoughts
        self.max_seq_len = max_seq_len

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        # ---- tokenizer ----
        print(f"[CoconutEngine] Loading tokenizer from {model_path} ...")
        self.tokenizer = self._init_tokenizer(model_path)

        self.bot_id = self.tokenizer.convert_tokens_to_ids(BOT_TOKEN)
        self.eot_id = self.tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        self.thought_id = self.tokenizer.convert_tokens_to_ids(THOUGHT_TOKEN)
        self._validate_special_tokens()

        # ---- model ----
        print(f"[CoconutEngine] Loading model from {model_path} ...")
        self.model = self._init_model(model_path, adapter_path, device, trust_remote_code)
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.embedding = self.model.get_input_embeddings()

        print(
            f"[CoconutEngine] Ready.  device={self.device}  dtype={self.dtype}  "
            f"num_thoughts={self.num_thoughts}  "
            f"vocab={len(self.tokenizer)}  "
            f"<bot>={self.bot_id}  <eot>={self.eot_id}  <thought>={self.thought_id}"
        )

    # ------------------------------------------------------------------
    #  Initialisation helpers
    # ------------------------------------------------------------------

    def _init_tokenizer(self, model_path: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        existing = set(tokenizer.additional_special_tokens or [])
        to_add = [t for t in [BOT_TOKEN, EOT_TOKEN, THOUGHT_TOKEN] if t not in existing]
        if to_add:
            all_special = list(existing) + to_add
            tokenizer.add_special_tokens({"additional_special_tokens": all_special})
            print(f"[CoconutEngine] Added special tokens to tokenizer: {to_add}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _init_model(
        self, model_path: str, adapter_path: Optional[str], device: str, trust_remote_code: bool
    ) -> AutoModelForCausalLM:
        device_map = device if device in ("auto", "cpu") else None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        # Merge LoRA adapters if provided
        if adapter_path:
            if not _HAS_PEFT:
                raise ImportError("peft is required to load LoRA adapters: pip install peft")
            print(f"[CoconutEngine] Merging LoRA adapters from {adapter_path} ...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print("[CoconutEngine] LoRA adapters merged successfully.")

        # Resize embeddings to accommodate newly added special tokens
        model.resize_token_embeddings(len(self.tokenizer))

        # Move to device if explicit GPU
        if device_map is None and device != "cpu" and torch.cuda.is_available():
            model = model.to(device)

        return model

    def _validate_special_tokens(self):
        unk = self.tokenizer.unk_token_id
        for name, tid in [("bot", self.bot_id), ("eot", self.eot_id), ("thought", self.thought_id)]:
            if tid is None or tid == unk:
                raise ValueError(
                    f"Special token <{name}> resolved to UNK / None.  "
                    f"Make sure the tokenizer was saved after adding special tokens."
                )

    # ------------------------------------------------------------------
    #  Core Coconut inference
    # ------------------------------------------------------------------

    def _build_prefix_ids(self, prompt: str) -> List[int]:
        """Tokenize prompt and wrap with ``<bot> <thought>*N <eot>``."""
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tokens + [self.bot_id] + [self.thought_id] * self.num_thoughts + [self.eot_id]

    @torch.inference_mode()
    def _coconut_prefill(
        self, input_ids: torch.Tensor
    ) -> tuple:
        """
        Run the Coconut iterative forward pass for the prefix.

        The sequence has the structure::

            [question tokens...] <bot> <thought> <thought> ... <thought> <eot>

        Processing happens in three phases:

        1. **Pre-thought pass** — a single forward pass over all tokens
           *before* the first ``<thought>`` (question + ``<bot>``).
        2. **Iterative thought passes** — for each ``<thought>`` position,
           the hidden state of the immediately preceding position replaces
           the thought token's embedding, then a single-token forward pass
           is executed (with KV-cache from prior positions).
        3. **Post-thought pass** — process any remaining tokens after the
           last thought (``<eot>``).

        Returns:
            ``(past_key_values, last_logits)`` where *last_logits* has
            shape ``[1, 1, vocab_size]``.
        """
        seq_len = input_ids.shape[1]
        inputs_embeds = self.embedding(input_ids)

        # Locate <thought> positions
        thought_positions = (input_ids[0] == self.thought_id).nonzero(as_tuple=True)[0].tolist()
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)

        # Fast path: no thoughts in the input
        if not thought_positions:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            return outputs.past_key_values, outputs.logits[:, -1:, :]

        first_thought_pos = thought_positions[0]

        # Phase 1 — process everything before the first thought
        outputs = self.model(
            inputs_embeds=inputs_embeds[:, :first_thought_pos, :],
            position_ids=position_ids[:, :first_thought_pos],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        kv_cache = outputs.past_key_values
        # Hidden state of the position immediately before the first thought
        # (this is the <bot> token).  It becomes the embedding for thought #0.
        prev_hidden = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]

        # Phase 2 — iterate through each thought token
        for pos in thought_positions:
            # Feed the previous hidden state as the input embedding
            outputs = self.model(
                inputs_embeds=prev_hidden,
                position_ids=position_ids[:, pos : pos + 1],
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values
            prev_hidden = outputs.hidden_states[-1][:, -1:, :]

        # Phase 3 — process everything after the last thought (typically <eot>)
        remaining_start = thought_positions[-1] + 1
        if remaining_start < seq_len:
            outputs = self.model(
                inputs_embeds=inputs_embeds[:, remaining_start:, :],
                position_ids=position_ids[:, remaining_start:],
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values

        return kv_cache, outputs.logits[:, -1:, :]

    # ------------------------------------------------------------------
    #  Token sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        generated_ids: List[int],
    ) -> int:
        """Apply temperature / top-k / top-p / repetition-penalty and sample."""
        logits = logits.clone()

        # Repetition penalty (Keskar et al., 2019)
        if repetition_penalty != 1.0 and generated_ids:
            prev = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
            penalty_logits = logits[0, prev]
            penalty_logits = torch.where(
                penalty_logits > 0,
                penalty_logits / repetition_penalty,
                penalty_logits * repetition_penalty,
            )
            logits[0, prev] = penalty_logits

        # Greedy
        if temperature <= 0 or temperature < 1e-7:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature

        # Top-k
        if 0 < top_k < logits.shape[-1]:
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(-1, topk_idx, topk_vals)

        # Top-p (nucleus)
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = (cumulative - torch.softmax(sorted_logits, dim=-1)) > top_p
            sorted_logits[mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    # ------------------------------------------------------------------
    #  Public generation API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a completion using the Coconut inference procedure.

        Args:
            prompt:             Raw text prompt (question / instruction).
            max_new_tokens:     Maximum number of answer tokens to generate.
            temperature:        Sampling temperature (0 = greedy decoding).
            top_p:              Nucleus sampling threshold.
            top_k:              Top-k filtering (<=0 = disabled).
            repetition_penalty: Multiplier for already-generated tokens.
            stop:               Optional list of stop strings.

        Returns:
            ``{"text": str, "tokens": int, "finish_reason": "stop"|"length"}``
        """
        prefix_ids = self._build_prefix_ids(prompt)
        if len(prefix_ids) > self.max_seq_len:
            prefix_ids = prefix_ids[: self.max_seq_len]
        prefix_len = len(prefix_ids)

        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

        # ---- Coconut prefill (latent thought processing) ----
        kv_cache, logits = self._coconut_prefill(input_ids)

        # ---- Autoregressive decoding ----
        generated_ids: List[int] = []
        current_logits = logits[:, -1, :]  # [1, vocab_size]
        finish_reason = "length"

        for step in range(max_new_tokens):
            next_id = self._sample_token(
                current_logits, temperature, top_p, top_k, repetition_penalty, generated_ids
            )

            if next_id == self.tokenizer.eos_token_id:
                finish_reason = "stop"
                break

            generated_ids.append(next_id)

            # Check stop sequences
            if stop:
                text_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                for seq in stop:
                    idx = text_so_far.find(seq)
                    if idx != -1:
                        return {
                            "text": text_so_far[:idx],
                            "tokens": len(generated_ids),
                            "finish_reason": "stop",
                        }

            # Forward pass for the newly generated token
            next_embed = self.embedding(
                torch.tensor([[next_id]], device=self.device, dtype=torch.long)
            )
            next_pos = torch.tensor(
                [[prefix_len + step]], device=self.device, dtype=torch.long
            )

            outputs = self.model(
                inputs_embeds=next_embed,
                position_ids=next_pos,
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values
            current_logits = outputs.logits[:, -1, :]

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {"text": text, "tokens": len(generated_ids), "finish_reason": finish_reason}


# ---------------------------------------------------------------------------
#  FastAPI server (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _build_server(engine: CoconutEngine, model_name: str) -> "FastAPI":
    """Create a FastAPI app with OpenAI-compatible endpoints."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI + Uvicorn are required for serving: pip install fastapi uvicorn")

    app = FastAPI(title="Coconut Inference Server")

    # ---- Pydantic request / response schemas ----

    class CompletionRequest(PydanticBaseModel):
        model: str = model_name
        prompt: str | List[str] = ""
        max_tokens: int = PydanticField(default=1024, alias="max_tokens")
        temperature: float = 0.0
        top_p: float = 1.0
        top_k: int = -1
        repetition_penalty: float = 1.0
        stop: Optional[List[str]] = None
        stream: bool = False

    class ChatMessage(PydanticBaseModel):
        role: str
        content: str

    class ChatCompletionRequest(PydanticBaseModel):
        model: str = model_name
        messages: List[ChatMessage]
        max_tokens: int = 1024
        temperature: float = 0.0
        top_p: float = 1.0
        top_k: int = -1
        repetition_penalty: float = 1.0
        stop: Optional[List[str]] = None
        stream: bool = False

    # ---- helpers ----

    def _messages_to_prompt(messages: List[ChatMessage]) -> str:
        """Flatten chat messages into a single prompt string."""
        parts: List[str] = []
        for msg in messages:
            parts.append(f"{msg.role.capitalize()}:\n{msg.content.strip()}")
        return "\n\n".join(parts) + "\n"

    def _make_completion_response(
        text: str, tokens: int, finish_reason: str, req_model: str
    ) -> Dict[str, Any]:
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req_model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens,
                "total_tokens": tokens,
            },
        }

    def _make_chat_response(
        text: str, tokens: int, finish_reason: str, req_model: str
    ) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens,
                "total_tokens": tokens,
            },
        }

    # ---- routes ----

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "coconut",
                }
            ],
        }

    @app.post("/v1/completions")
    async def create_completion(req: CompletionRequest):
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        choices = []
        total_tokens = 0
        for i, prompt in enumerate(prompts):
            result = engine.generate(
                prompt=prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                stop=req.stop,
            )
            choices.append(
                {
                    "index": i,
                    "text": result["text"],
                    "logprobs": None,
                    "finish_reason": result["finish_reason"],
                }
            )
            total_tokens += result["tokens"]

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(req: ChatCompletionRequest):
        prompt = _messages_to_prompt(req.messages)
        result = engine.generate(
            prompt=prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
        )
        return _make_chat_response(result["text"], result["tokens"], result["finish_reason"], req.model)

    return app


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Coconut Inference Server — drop-in replacement for `vllm serve`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model", type=str, required=True, help="Path or HF hub ID of the merged model.")
    p.add_argument("--adapter", type=str, default=None, help="Optional path to LoRA adapter dir.")
    p.add_argument("--num-thoughts", type=int, default=12, help="Number of <thought> tokens.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda", help="Device: cuda, cpu, or auto.")
    p.add_argument("--max-seq-len", type=int, default=8192, help="Maximum total sequence length.")
    p.add_argument("--trust-remote-code", action="store_true", default=True)

    # Server
    p.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    p.add_argument("--port", type=int, default=8000, help="Server port.")

    # Quick single-prompt test (no server)
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="If set, run a single inference instead of starting the server.",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()

    engine = CoconutEngine(
        model_path=args.model,
        adapter_path=args.adapter,
        num_thoughts=args.num_thoughts,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_seq_len=args.max_seq_len,
    )

    # Quick single-prompt mode
    if args.prompt is not None:
        print("\n" + "=" * 70)
        print("PROMPT:")
        print(args.prompt)
        print("-" * 70)

        t0 = time.time()
        result = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0

        print("OUTPUT:")
        print(result["text"])
        print("-" * 70)
        print(
            f"tokens={result['tokens']}  "
            f"finish_reason={result['finish_reason']}  "
            f"time={elapsed:.2f}s  "
            f"tok/s={result['tokens'] / max(elapsed, 1e-6):.1f}"
        )
        print("=" * 70)
        return

    # Server mode
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI and Uvicorn are required for server mode.\n"
            "Install them with:  pip install fastapi uvicorn"
        )

    app = _build_server(engine, model_name=args.model)
    print(f"\n[CoconutEngine] Starting server on http://{args.host}:{args.port}")
    print(f"[CoconutEngine] OpenAI-compatible API at http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
