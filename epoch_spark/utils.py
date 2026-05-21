"""Shared utilities for Epoch-Spark PoC."""

import sys
import os
import torch
import time
from pathlib import Path

from config import MODEL_PATH, MASK_ID, EOS_ID, BLOCK_LENGTH


def get_device(gpu_id=0):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


_vllm_config_ctx = None


def _ensure_vllm_env(device="cuda:0"):
    """Initialize vLLM distributed env and config context (idempotent)."""
    global _vllm_config_ctx
    import socket
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        import types, importlib.util
        sys.modules.pop("deep_ep", None)
        _fake = types.ModuleType("deep_ep")
        _fake.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None)
        _fake.__path__ = []
        # Stubs for class-level type annotations in vllm
        _fake.Buffer = type("Buffer", (), {
            "get_dispatch_config": staticmethod(lambda *a, **kw: None),
            "get_combine_config": staticmethod(lambda *a, **kw: None),
        })
        _fake.Config = type("Config", (), {})
        _fake.EventOverlap = type("EventOverlap", (), {})
        sys.modules["deep_ep"] = _fake

    torch.cuda.set_device(device)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    vcfg = VllmConfig(parallel_config=pcfg)

    if _vllm_config_ctx is None:
        _vllm_config_ctx = set_current_vllm_config(vcfg)
        _vllm_config_ctx.__enter__()

    if not torch.distributed.is_initialized():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
        distributed.initialize_model_parallel(1, backend="nccl")


def load_model_and_tokenizer(device="cuda:0", dtype=torch.bfloat16):
    """Load LLaDA2.0-mini model and tokenizer onto specified device.

    Uses the same loading pattern as existing Epoch experiments:
    vLLM config context + dInfer's custom load_state_dict which packs
    expert weights into FusedMoE's w13/w2 format.
    """
    _ensure_vllm_env(device)

    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )

    # Clear FusedMoE's global layer name registry to allow re-instantiation
    try:
        from vllm.config.vllm import get_current_vllm_config
        vconfig = get_current_vllm_config()
        cc = vconfig.compilation_config
        if hasattr(cc, "static_forward_context"):
            cc.static_forward_context.clear()
        if hasattr(cc, "static_all_moe_layers"):
            cc.static_all_moe_layers.clear()
    except Exception:
        pass

    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(MODEL_PATH, torch_dtype=dtype, device=device)
    model = model.to(device)

    # Replace FusedMoE forward with direct expert computation
    # This bypasses vLLM's forward context requirements and gives us
    # direct control over expert weight access (needed for residency manager)
    _patch_fused_moe_to_direct(model)

    # Warmup
    with torch.inference_mode():
        warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
        _ = model(warmup_tok, use_cache=False)

    return model, tokenizer, config


def _fused_moe_forward(self, hidden_states, router_logits):
    """Fused MoE forward using vLLM's Triton kernel.

    Routes tokens through grouped top-k, then executes via fused_experts_impl.
    Stores routing info for downstream hooks/controllers.
    """
    from triton_moe import fused_experts, grouped_topk

    topk_weights, topk_ids = grouped_topk(router_logits)

    output = fused_experts(
        hidden_states, self.w13_weight, self.w2_weight,
        topk_weights, topk_ids,
    )

    self._last_topk_ids = topk_ids
    self._last_topk_weights = topk_weights
    self._last_active_experts = topk_ids.unique().tolist()

    return output


def _patch_fused_moe_to_direct(model):
    """Replace FusedMoE.forward with fused Triton expert computation."""
    from vllm.model_executor.layers.fused_moe import FusedMoE
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, FusedMoE):
            mod.forward = lambda *a, _mod=mod, **kw: _fused_moe_forward(
                _mod,
                kw.get("hidden_states", a[0] if a else None),
                kw.get("router_logits", a[1] if len(a) > 1 else None),
            )
            count += 1
    print(f"[patch] Replaced {count} FusedMoE modules with fused Triton forward")


_cached_vllm_cfg = None


def make_forward_context(num_tokens):
    """Create a vLLM forward context for standalone FusedMoE execution."""
    global _cached_vllm_cfg
    from vllm.forward_context import set_forward_context

    if _cached_vllm_cfg is None:
        from vllm.config import VllmConfig, ModelConfig, CacheConfig
        _llama_path = "/mnt/models/Meta-Llama-3-8B-Instruct"
        _cached_vllm_cfg = VllmConfig(
            model_config=ModelConfig(
                model=_llama_path, tokenizer=_llama_path,
                dtype=torch.bfloat16, seed=42, revision=None,
            ),
            cache_config=CacheConfig(
                block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto",
            ),
        )
    return set_forward_context(
        attn_metadata=None, vllm_config=_cached_vllm_cfg, num_tokens=num_tokens,
    )


def prepare_input(tokenizer, prompt, gen_length=256, device="cuda:0"):
    """Tokenize prompt and pad generation region with MASK tokens."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # [1, L]
    prompt_len = input_ids.shape[1]
    mask_tokens = torch.full(
        (1, gen_length), MASK_ID, dtype=torch.long, device=device
    )
    x = torch.cat([input_ids, mask_tokens], dim=1)  # [1, L + gen_length]
    return x, prompt_len


def threshold_decode(logits, x, mask_id=MASK_ID, temperature=0.0):
    """Simple confidence-based threshold decode for one iteration.

    Returns updated x with some MASK positions committed, and number of newly decoded tokens.
    """
    live_mask = (x == mask_id)  # [B, S]
    if not live_mask.any():
        return x, 0

    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)

    pred_tokens = logits.argmax(dim=-1)  # [B, S]
    confidence = probs.max(dim=-1).values  # [B, S]

    confidence_at_mask = confidence.clone()
    confidence_at_mask[~live_mask] = -1.0

    n_mask = live_mask.sum().item()
    n_to_decode = max(1, n_mask // BLOCK_LENGTH) if n_mask > 0 else 0

    if n_to_decode > 0 and n_mask > 0:
        flat_conf = confidence_at_mask.view(-1)
        _, top_idx = flat_conf.topk(min(n_to_decode, n_mask))
        flat_x = x.view(-1)
        flat_pred = pred_tokens.view(-1)
        flat_x[top_idx] = flat_pred[top_idx]
        x = flat_x.view(x.shape)

    newly_decoded = n_to_decode
    return x, newly_decoded


def get_num_transfer_tokens(mask_count, steps):
    """LLaDA-style: distribute mask_count tokens across steps."""
    base = mask_count // steps
    remainder = mask_count % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


def gpu_mem_mb(device=0):
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated(device) / (1024 * 1024)


def gpu_mem_reserved_mb(device=0):
    """Current GPU memory reserved in MB."""
    return torch.cuda.memory_reserved(device) / (1024 * 1024)


class Timer:
    """Simple timer context manager."""
    def __init__(self, name="", sync_cuda=True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.elapsed_ms = 0

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed_ms:.2f} ms")
