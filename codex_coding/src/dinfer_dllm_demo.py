#!/usr/bin/env python3
"""
Minimal dInfer-style diffusion LLM demo.

This script mirrors the simplest dInfer path:
block-wise decoding with threshold-based parallel unmasking.

It is intentionally self-contained so it can run in environments where the
top-level `dinfer` package import is blocked by missing optional backends
such as `vllm` or `sglang`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal dInfer-style diffusion LLM demo and print timing stats."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local path or Hugging Face repo id of the diffusion LLM checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short introduction to the Great Wall.",
        help="User prompt for generation.",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=64,
        help="Maximum generated token slots in the diffusion region.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Block size for block-wise diffusion decoding.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold used by threshold parallel decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Gumbel sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--mask-id",
        type=int,
        default=None,
        help="Override mask token id. If omitted, infer from tokenizer/model name.",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=None,
        help="Override EOS token id. If omitted, infer from tokenizer/model name.",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap the prompt with the tokenizer chat template when available.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model/tokenizer from local cache or local path.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print per-step denoising progress.",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help="Print one trace line every N denoising steps.",
    )
    parser.add_argument(
        "--show-partial-text",
        action="store_true",
        help="Decode and print the current partial output in trace mode.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional JSON path for saving timing and generation metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit without loading the model.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_special_ids(
    model_path: str,
    tokenizer: Any,
    mask_id: int | None,
    eos_id: int | None,
) -> tuple[int, int]:
    if mask_id is None:
        mask_id = getattr(tokenizer, "mask_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)

    path_lower = model_path.lower()
    if mask_id is None:
        if "llada2" in path_lower or "llada-moe" in path_lower or "flash" in path_lower:
            mask_id = 156895
        elif "llada" in path_lower:
            mask_id = 126336
        elif "sdar" in path_lower:
            mask_id = 151669

    if eos_id is None:
        if "llada2" in path_lower or "llada-moe" in path_lower or "flash" in path_lower:
            eos_id = 156892
        elif "llada" in path_lower:
            eos_id = 126081

    if mask_id is None or eos_id is None:
        raise ValueError(
            "Failed to infer mask/eos ids automatically. Please pass --mask-id and --eos-id."
        )

    return int(mask_id), int(eos_id)


def build_prompt_ids(
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool,
    device: torch.device,
) -> torch.Tensor:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"].to(device)


class TokenArray:
    def __init__(
        self,
        prompt: torch.Tensor,
        gen_length: int,
        mask_id: int,
        eos_id: int,
        device: torch.device,
    ) -> None:
        self.prompt = prompt.to(device)
        self.data = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        self.data[:, : prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length
        self.mask_id = mask_id
        self.eos_id = eos_id

    @property
    def total_length(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, idx: Any) -> torch.Tensor:
        return self.data[idx]

    def __setitem__(self, idx: Any, value: torch.Tensor) -> None:
        self.data[idx] = value


@dataclass
class BlockLoc:
    start: int
    end: int


class BlockIterator:
    def __init__(self, x: TokenArray, block_length: int) -> None:
        self.x = x
        self.block_length = block_length
        self._iter = 0
        self.first_block_start = self.x.prompt.shape[1]

    def __iter__(self) -> "BlockIterator":
        self._iter = 0
        return self

    def __next__(self) -> tuple[BlockLoc, torch.Tensor]:
        start = self.first_block_start + self._iter * self.block_length
        if start >= self.x.total_length:
            raise StopIteration
        end = min(start + self.block_length, self.x.total_length)
        self._iter += 1
        return BlockLoc(start=start, end=end), self.x[:, start:end]


class BlockIteratorFactory:
    def create(self, x: TokenArray, block_length: int) -> BlockIterator:
        return BlockIterator(x, block_length)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if math.isclose(temperature, 0.0):
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index_threshold(
    logits: torch.Tensor,
    temperature: float,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    mask_id: int,
    threshold: float,
    use_float64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if use_float64:
        probs = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        probs = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

    actual_threshold = torch.max(confidence, dim=1)[0].sub(1e-5).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index


class ThresholdParallelDecoder:
    def __init__(
        self,
        temperature: float,
        threshold: float,
        mask_id: int,
        eos_id: int,
        use_float64: bool = False,
    ) -> None:
        self.temperature = temperature
        self.threshold = threshold
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.use_float64 = use_float64

    def block_init(self, block_x: torch.Tensor, block_id: int) -> None:
        del block_x, block_id

    def decode(
        self,
        logits: torch.Tensor,
        block_start: int,
        block_end: int,
        x: TokenArray,
    ) -> int:
        mask_index = x[:, block_start:block_end] == self.mask_id
        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(
            logits=logits,
            temperature=self.temperature,
            mask_index=mask_index,
            x=curr_x,
            mask_id=self.mask_id,
            threshold=self.threshold,
            use_float64=self.use_float64,
        )
        transfer_index = transfer_index & mask_index
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        return int(transfer_index.sum().item())


class TraceBlockWiseDiffusionLLM:
    def __init__(
        self,
        model: torch.nn.Module,
        decoder: ThresholdParallelDecoder,
        tokenizer: Any,
        trace: bool = False,
        trace_every: int = 1,
        show_partial_text: bool = False,
        early_stop: bool = True,
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.trace = trace
        self.trace_every = max(1, trace_every)
        self.show_partial_text = show_partial_text
        self.early_stop = early_stop
        self.iterator_factory = BlockIteratorFactory()

    def _trim_to_eos(self, seq: torch.Tensor) -> torch.Tensor:
        eos_pos = (seq == self.decoder.eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            return seq[: int(eos_pos[0].item())]
        return seq

    def _partial_text(self, seq: torch.Tensor, prompt_len: int) -> str:
        trimmed = self._trim_to_eos(seq)
        generated = trimmed[prompt_len:]
        if generated.numel() == 0:
            return ""
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return text.replace("\n", "\\n")

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int,
        block_length: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = TokenArray(
            prompt=prompt_ids,
            gen_length=gen_length,
            mask_id=self.decoder.mask_id,
            eos_id=self.decoder.eos_id,
            device=prompt_ids.device,
        )
        prompt_len = prompt_ids.shape[1]
        total_forwards = 0
        accepted_total = 0
        blocks_finished = 0

        for block_id, (block_loc, block) in enumerate(self.iterator_factory.create(x, block_length)):
            self.decoder.block_init(block, block_id)
            step_in_block = 0
            while (x[:, block_loc.start:block_loc.end] == self.decoder.mask_id).sum() > 0:
                logits = self.model(x.data).logits[:, block_loc.start:block_loc.end]
                accepted = self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                total_forwards += 1
                accepted_total += accepted
                step_in_block += 1

                remaining = int((x[:, block_loc.start:block_loc.end] == self.decoder.mask_id).sum().item())
                if self.trace and step_in_block % self.trace_every == 0:
                    msg = (
                        f"[block {block_id:02d} step {step_in_block:02d}] "
                        f"accepted={accepted:02d} remaining={remaining:02d}"
                    )
                    if self.show_partial_text:
                        msg += f" | partial={self._partial_text(x.data[0], prompt_len)}"
                    print(msg, flush=True)

            blocks_finished += 1
            if self.early_stop and torch.any(x[:, block_loc.start:block_loc.end] == self.decoder.eos_id):
                x[:, block_loc.end:] = self.decoder.eos_id
                break

        full_output = self._trim_to_eos(x.data[0]).clone()
        generated_ids = full_output[prompt_len:]
        stats = {
            "num_forwards": total_forwards,
            "accepted_total": accepted_total,
            "blocks_finished": blocks_finished,
            "generated_tokens": int(generated_ids.numel()),
            "avg_tokens_per_forward": (
                float(generated_ids.numel()) / float(total_forwards) if total_forwards > 0 else 0.0
            ),
        }
        return full_output.unsqueeze(0), stats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    config_preview = {
        "model_path": args.model_path,
        "device": str(device),
        "dtype": args.dtype,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "temperature": args.temperature,
    }

    if args.dry_run:
        print(json.dumps(config_preview, ensure_ascii=False, indent=2))
        return

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    mask_id, eos_id = infer_special_ids(
        model_path=args.model_path,
        tokenizer=tokenizer,
        mask_id=args.mask_id,
        eos_id=args.eos_id,
    )

    prompt_ids = build_prompt_ids(
        tokenizer=tokenizer,
        prompt=args.prompt,
        use_chat_template=args.use_chat_template,
        device=device,
    )

    print("Loading model...", flush=True)
    model_dtype = dtype if device.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)

    decoder = ThresholdParallelDecoder(
        temperature=args.temperature,
        threshold=args.threshold,
        mask_id=mask_id,
        eos_id=eos_id,
        use_float64=True,
    )
    engine = TraceBlockWiseDiffusionLLM(
        model=model,
        decoder=decoder,
        tokenizer=tokenizer,
        trace=args.trace,
        trace_every=args.trace_every,
        show_partial_text=args.show_partial_text,
        early_stop=True,
    )

    print(
        f"Resolved ids: mask_id={mask_id}, eos_id={eos_id}, prompt_len={prompt_ids.shape[1]}",
        flush=True,
    )

    sync_device(device)
    start = time.perf_counter()
    output_ids, stats = engine.generate(
        prompt_ids=prompt_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
    )
    sync_device(device)
    elapsed = time.perf_counter() - start

    generated_text = tokenizer.decode(
        output_ids[0, prompt_ids.shape[1] :],
        skip_special_tokens=True,
    )

    metrics = {
        "model_path": args.model_path,
        "device": str(device),
        "dtype": str(model_dtype).replace("torch.", ""),
        "mask_id": mask_id,
        "eos_id": eos_id,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "generated_tokens": stats["generated_tokens"],
        "num_forwards": stats["num_forwards"],
        "blocks_finished": stats["blocks_finished"],
        "elapsed_sec": elapsed,
        "tokens_per_sec": (stats["generated_tokens"] / elapsed) if elapsed > 0 else 0.0,
        "avg_tokens_per_forward": stats["avg_tokens_per_forward"],
        "prompt": args.prompt,
        "generated_text": generated_text,
    }

    print("\n=== Generation ===")
    print(generated_text)
    print("\n=== Metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.metrics_output:
        output_path = Path(args.metrics_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
