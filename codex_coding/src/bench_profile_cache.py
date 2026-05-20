#!/usr/bin/env python3
"""
Profile cache vs no-cache forward path — operator-level bottleneck analysis.

Scenarios:
  S1: no-cache,  batch=1,  long prompt  (baseline)
  S2: cache-opt, batch=1,  long prompt  (gap analysis)
  S3: cache-opt, batch=8,  long prompt  (crossover)
  S4: cache-opt, batch=32, long prompt  (serving target)

Approach: monkey-patch key functions with cuda events (no mid-execution sync).
All events are resolved after generate() completes.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

# === Config ===
REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"

MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
GEN_LENGTH = 128
BLOCK_LENGTH = 32
THRESHOLD = 0.90
MASK_ID = 156895
EOS_ID = 156892
DEVICE = "cuda:0"

LONG_PROMPT = """Please solve the following problems step by step.

Problem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?

Problem 2: A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than twice the width, find the dimensions of the garden.

Problem 3: In a class of 40 students, 25 study Mathematics, 20 study Physics, and 10 study both subjects. How many students study neither Mathematics nor Physics?

Problem 4: A cone has a radius of 7 cm and a slant height of 25 cm. Calculate the total surface area and the volume of the cone.

Problem 5: A bank offers compound interest at 8% per annum, compounded quarterly. If you deposit $5000, how much will you have after 3 years?

Problem 6: Two pipes can fill a tank. Pipe A fills the tank in 12 hours and Pipe B fills it in 18 hours. If both pipes are opened together, but Pipe B is closed after 4 hours, how long will it take Pipe A alone to fill the remaining tank?

Problem 7: A sequence is defined as follows: a(1) = 2, a(2) = 5, and for n >= 3, a(n) = 2*a(n-1) - a(n-2) + 3. Find the first 8 terms.

Problem 8: A factory produces widgets on two assembly lines. Line A produces 300 widgets per hour with a defect rate of 2%. Line B produces 200 widgets per hour with a defect rate of 1.5%. If the factory runs both lines for 8 hours, what is the overall defect rate?

Problem 9: A cylindrical water tank with radius 3 meters and height 10 meters is being filled at a rate of 2 cubic meters per minute while being drained at 0.5 cubic meters per minute. How long will it take to fill completely?"""


# ============================================================
# CudaTimer — no sync during execution, resolve at the end
# ============================================================
class CudaTimer:
    def __init__(self):
        self.events = []  # (label, start_event, end_event)
        self._stack = []  # indices of events whose stop() hasn't been called yet

    def start(self, label):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        idx = len(self.events)
        self.events.append((label, s, e))
        self._stack.append(idx)

    def stop(self):
        idx = self._stack.pop()
        self.events[idx][2].record()

    def reset(self):
        self.events = []
        self._stack = []

    def resolve(self):
        """Sync GPU and return {label: [elapsed_ms, ...]}."""
        torch.cuda.synchronize()
        records = defaultdict(list)
        for label, s, e in self.events:
            records[label].append(s.elapsed_time(e))
        return dict(records)


timer = CudaTimer()


# ============================================================
# Monkey-patching
# ============================================================
_originals = {}
_hooks_installed = False


def install_hooks(model):
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    from dinfer.decoding.parallel_strategy import ThresholdParallelDecoder
    from dinfer.decoding.utils import (
        BlockDiffusionPrefixCacheManager,
        DiffusionKVCacheManager,
        KVCache,
    )
    from dinfer.decoding.generate_uniform import BlockDiffusionRunner

    # 1. model.forward
    _originals["model_forward"] = model.forward
    def _tf(*a, **kw):
        timer.start("model_forward")
        r = _originals["model_forward"](*a, **kw)
        timer.stop()
        return r
    model.forward = _tf

    # 2. ThresholdParallelDecoder.decode
    _originals["decoder_decode"] = ThresholdParallelDecoder.decode
    def _td(self, *a, **kw):
        timer.start("decoder_decode")
        r = _originals["decoder_decode"](self, *a, **kw)
        timer.stop()
        return r
    ThresholdParallelDecoder.decode = _td

    # 3. BlockDiffusionPrefixCacheManager.extend_cache
    _originals["cache_extend"] = BlockDiffusionPrefixCacheManager.extend_cache
    def _te(self, *a, **kw):
        timer.start("cache_extend")
        r = _originals["cache_extend"](self, *a, **kw)
        timer.stop()
        return r
    BlockDiffusionPrefixCacheManager.extend_cache = _te

    # 4. DiffusionKVCacheManager.update (includes consolidate inside)
    _originals["cache_mgr_update"] = DiffusionKVCacheManager.update
    def _tu(self, *a, **kw):
        timer.start("cache_mgr_update")
        r = _originals["cache_mgr_update"](self, *a, **kw)
        timer.stop()
        return r
    DiffusionKVCacheManager.update = _tu

    # 5. KVCache.consolidate (inner portion of update)
    _originals["kvcache_consolidate"] = KVCache.consolidate
    def _tc(self, *a, **kw):
        timer.start("kvcache_consolidate")
        r = _originals["kvcache_consolidate"](self, *a, **kw)
        timer.stop()
        return r
    KVCache.consolidate = _tc

    # 6. BlockDiffusionRunner.prefill
    _originals["runner_prefill"] = BlockDiffusionRunner.prefill
    def _tp(self, *a, **kw):
        timer.start("prefill_total")
        r = _originals["runner_prefill"](self, *a, **kw)
        timer.stop()
        return r
    BlockDiffusionRunner.prefill = _tp


def remove_hooks(model):
    global _hooks_installed
    if not _hooks_installed:
        return
    _hooks_installed = False

    from dinfer.decoding.parallel_strategy import ThresholdParallelDecoder
    from dinfer.decoding.utils import (
        BlockDiffusionPrefixCacheManager,
        DiffusionKVCacheManager,
        KVCache,
    )
    from dinfer.decoding.generate_uniform import BlockDiffusionRunner

    model.forward = _originals["model_forward"]
    ThresholdParallelDecoder.decode = _originals["decoder_decode"]
    BlockDiffusionPrefixCacheManager.extend_cache = _originals["cache_extend"]
    DiffusionKVCacheManager.update = _originals["cache_mgr_update"]
    KVCache.consolidate = _originals["kvcache_consolidate"]
    BlockDiffusionRunner.prefill = _originals["runner_prefill"]


# ============================================================
# Helpers
# ============================================================
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def trim_after_eos(t, eos_id):
    p = (t == eos_id).nonzero(as_tuple=True)[0]
    return t[: int(p[0].item())] if p.numel() > 0 else t


def print_breakdown(records, wall_ms, meta):
    """Print operator breakdown table."""
    # Leaf operations (non-overlapping on GPU stream):
    #   model_forward, decoder_decode, cache_extend, cache_mgr_update
    # Note: kvcache_consolidate is INSIDE cache_mgr_update (subset, not additive)
    # Note: prefill_total is an AGGREGATE containing model_forward + cache_mgr_update

    leaf_keys = ["model_forward", "decoder_decode", "cache_extend", "cache_mgr_update"]
    info_keys = ["kvcache_consolidate", "prefill_total"]

    print(f"\n  {'Operation':<28s} {'Total(ms)':>10s} {'Count':>6s} {'Avg(ms)':>10s} {'%wall':>7s}")
    print(f"  {'-'*28} {'-'*10} {'-'*6} {'-'*10} {'-'*7}")

    leaf_total = 0.0
    for key in leaf_keys:
        times = records.get(key, [])
        if not times:
            continue
        total = sum(times)
        count = len(times)
        avg = total / count
        pct = total / wall_ms * 100
        leaf_total += total
        print(f"  {key:<28s} {total:>10.1f} {count:>6d} {avg:>10.2f} {pct:>6.1f}%")

    other = wall_ms - leaf_total
    pct_other = other / wall_ms * 100
    print(f"  {'--- CPU/Python/other ---':<28s} {other:>10.1f} {'':>6s} {'':>10s} {pct_other:>6.1f}%")
    print(f"  {'=== TOTAL WALL ===':<28s} {wall_ms:>10.1f}")

    # Info rows (not additive)
    if any(records.get(k) for k in info_keys):
        print(f"\n  (info, not additive):")
        for key in info_keys:
            times = records.get(key, [])
            if not times:
                continue
            total = sum(times)
            count = len(times)
            avg = total / count
            print(f"  {key:<28s} {total:>10.1f} {count:>6d} {avg:>10.2f}")

    # model_forward time distribution (first call = prefill for cache path)
    mf_times = records.get("model_forward", [])
    if len(mf_times) > 1:
        print(f"\n  model_forward distribution:")
        print(f"    first call (prefill?):  {mf_times[0]:.2f} ms")
        rest = mf_times[1:]
        print(f"    remaining {len(rest)} calls:    avg={sum(rest)/len(rest):.2f}  min={min(rest):.2f}  max={max(rest):.2f} ms")


def run_scenario(label, dllm_factory, input_ids, batch_size, device, model):
    """Warmup + 2 profiled runs, return median."""
    batched = input_ids.repeat(batch_size, 1)
    plen = input_ids.shape[1]

    print(f"\n{'='*85}")
    print(f"  {label}  (batch={batch_size}, prompt={plen}, gen={GEN_LENGTH}, block={BLOCK_LENGTH})")
    print(f"{'='*85}")

    # Warmup (no hooks)
    print("  Warmup ...", flush=True)
    remove_hooks(model)
    dllm = dllm_factory()
    with torch.inference_mode():
        dllm.generate(batched, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
    torch.cuda.empty_cache()

    # Profiled runs
    install_hooks(model)
    all_runs = []

    for r_idx in range(2):
        timer.reset()
        dllm = dllm_factory()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            out = dllm.generate(batched, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - t0
        wall_ms = wall_s * 1000.0

        records = timer.resolve()

        fwd_count = len(records.get("model_forward", []))
        ttok = sum(
            int(trim_after_eos(out[b][plen:], EOS_ID).numel())
            for b in range(batch_size)
        )

        print(f"\n  Run {r_idx+1}: {ttok} tok, {fwd_count} model_fwd calls, "
              f"{wall_ms:.0f}ms, {ttok/wall_s:.0f} tok/s, {fwd_count/wall_s:.1f} fwd/s")
        print_breakdown(records, wall_ms, {"batch": batch_size})

        all_runs.append({
            "wall_ms": wall_ms,
            "fwd_count": fwd_count,
            "tok": ttok,
            "tok_s": ttok / wall_s,
            "fwd_s": fwd_count / wall_s,
            "records": {k: {"total": sum(v), "count": len(v), "avg": sum(v)/len(v),
                            "times": v}
                        for k, v in records.items()},
        })

    remove_hooks(model)
    torch.cuda.empty_cache()

    # Return the run closest to median fwd/s
    fps_list = [r["fwd_s"] for r in all_runs]
    med_val = sorted(fps_list)[len(fps_list) // 2]
    med_idx = min(range(len(fps_list)), key=lambda i: abs(fps_list[i] - med_val))
    best = all_runs[med_idx]
    best["label"] = label
    best["batch_size"] = batch_size
    best["prompt_len"] = plen
    return best


# ============================================================
# Main
# ============================================================
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLM,
        BlockDiffusionLLMAttnmask,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    cfg = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            _ = model(
                torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                use_cache=False,
            )

        def make_decoder():
            return ThresholdParallelDecoder(
                temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID
            )

        prompt_text = LONG_PROMPT
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        long_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        print(f"Long prompt: {long_ids.shape[1]} tokens\n")

        results = {}

        # S1: no-cache, batch=1
        results["S1"] = run_scenario(
            "S1: no-cache b=1",
            lambda: BlockDiffusionLLMAttnmask(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                early_stop=True,
            ),
            long_ids, batch_size=1, device=device, model=model,
        )

        # S2: cache-opt, batch=1
        results["S2"] = run_scenario(
            "S2: cache-opt b=1",
            lambda: BlockDiffusionLLM(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
                lazy_cache_update=True,
                inplace_cache_update=True,
            ),
            long_ids, batch_size=1, device=device, model=model,
        )

        # S3: cache-opt, batch=8
        results["S3"] = run_scenario(
            "S3: cache-opt b=8",
            lambda: BlockDiffusionLLM(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
                lazy_cache_update=True,
                inplace_cache_update=True,
            ),
            long_ids, batch_size=8, device=device, model=model,
        )

        # S4: cache-opt, batch=32
        results["S4"] = run_scenario(
            "S4: cache-opt b=32",
            lambda: BlockDiffusionLLM(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
                lazy_cache_update=True,
                inplace_cache_update=True,
            ),
            long_ids, batch_size=32, device=device, model=model,
        )

        # ============================================================
        # Cross-scenario comparison
        # ============================================================
        print(f"\n{'#'*95}")
        print(f"  CROSS-SCENARIO COMPARISON")
        print(f"{'#'*95}")

        def _get(res, op):
            return res.get("records", {}).get(op, {})

        hdr = (f"  {'Scenario':<22s} {'Wall':>7s} {'Fwds':>5s} {'tok/s':>7s} {'fwd/s':>6s}"
               f" │ {'model_fwd':>10s} {'decoder':>8s} {'extend':>8s} {'update':>8s}"
               f" {'consol':>8s} {'prefill':>8s} {'other':>8s}")
        print(hdr)
        print(f"  {'-'*22} {'-'*7} {'-'*5} {'-'*7} {'-'*6}"
              f" │ {'-'*10} {'-'*8} {'-'*8} {'-'*8}"
              f" {'-'*8} {'-'*8} {'-'*8}")

        for key in ["S1", "S2", "S3", "S4"]:
            r = results.get(key)
            if r is None:
                continue
            mf = _get(r, "model_forward")
            dd = _get(r, "decoder_decode")
            ce = _get(r, "cache_extend")
            cu = _get(r, "cache_mgr_update")
            co = _get(r, "kvcache_consolidate")
            pf = _get(r, "prefill_total")

            leaf_total = mf.get("total",0) + dd.get("total",0) + ce.get("total",0) + cu.get("total",0)
            other = r["wall_ms"] - leaf_total

            print(
                f"  {r['label']:<22s}"
                f" {r['wall_ms']:>7.0f} {r['fwd_count']:>5d} {r['tok_s']:>7.0f} {r['fwd_s']:>6.1f}"
                f" │ {mf.get('total',0):>10.0f} {dd.get('total',0):>8.0f}"
                f" {ce.get('total',0):>8.1f} {cu.get('total',0):>8.1f}"
                f" {co.get('total',0):>8.1f} {pf.get('total',0):>8.1f}"
                f" {other:>8.0f}"
            )

        # S1 vs S2 gap analysis
        s1 = results.get("S1")
        s2 = results.get("S2")
        if s1 and s2:
            print(f"\n  --- S1 vs S2 (no-cache vs cache, batch=1) ---")
            s1_mf_avg = _get(s1, "model_forward").get("avg", 0)
            s2_mf = _get(s2, "model_forward")
            s2_mf_times = s2_mf.get("times", [])
            s2_mf_prefill = s2_mf_times[0] if s2_mf_times else 0
            s2_mf_decode_avg = sum(s2_mf_times[1:]) / max(len(s2_mf_times)-1, 1) if len(s2_mf_times) > 1 else 0

            print(f"  S1 model_fwd avg:          {s1_mf_avg:.2f} ms  (all {_get(s1,'model_forward').get('count',0)} are decode)")
            print(f"  S2 model_fwd prefill:      {s2_mf_prefill:.2f} ms  (1 call)")
            print(f"  S2 model_fwd decode avg:   {s2_mf_decode_avg:.2f} ms  ({max(len(s2_mf_times)-1,0)} calls)")
            print(f"  S2 decoder_decode avg:     {_get(s2,'decoder_decode').get('avg',0):.2f} ms")
            print(f"  S2 cache_extend total:     {_get(s2,'cache_extend').get('total',0):.1f} ms  ({_get(s2,'cache_extend').get('count',0)} calls)")
            print(f"  S2 cache_mgr_update total: {_get(s2,'cache_mgr_update').get('total',0):.1f} ms  ({_get(s2,'cache_mgr_update').get('count',0)} calls)")

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "profile_cache_results.json"

    # Strip raw times lists for compact JSON (keep totals/avgs)
    json_out = {}
    for key, val in results.items():
        r_copy = {k: v for k, v in val.items() if k != "records"}
        r_copy["records"] = {}
        for op, data in val.get("records", {}).items():
            r_copy["records"][op] = {k: v for k, v in data.items() if k != "times"}
        json_out[key] = r_copy

    out_path.write_text(json.dumps(json_out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
