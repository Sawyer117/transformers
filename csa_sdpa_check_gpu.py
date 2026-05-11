"""
Compare two CSA per-query sparse-attention expressions on GPU, using SDPA
(F.scaled_dot_product_attention) as the attention backend instead of eager
matmul + softmax.

Custom additive masks force PyTorch SDPA away from the flash-attn fastpath;
the question this script answers is whether the mem-efficient backend can
tile through the `[B, H, S, S*k]` attention so #45892's gather + 5D mask
expression avoids the eager-path OOM at long S.

  Path A — #45892 (gather + 5D diagonal mask, viewed as [B, 1, S, S*k]):
      F.scaled_dot_product_attention(q, K_gathered, V_gathered, attn_mask=block_bias)
  Path B — scatter-bias on un-gathered compressed_kv:
      F.scaled_dot_product_attention(q, K_full,     V_full,     attn_mask=bias)

Sink-token logic is *omitted* from both paths because SDPA's API doesn't
expose a clean way to fold per-head learnable sinks into the softmax. The
two paths are still bit-equivalent to each other under SDPA — sink would
add the same constant offset to either side.

Usage:
    python csa_sdpa_check_gpu.py                          # auto-pick backend
    python csa_sdpa_check_gpu.py --sdpa-backend efficient # force mem-eff
    python csa_sdpa_check_gpu.py --sdpa-backend math      # eager-equivalent
    python csa_sdpa_check_gpu.py --mode correctness
    python csa_sdpa_check_gpu.py --mode speed
"""

import argparse
import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False


# ---------------------------------------------------------------------------
# Path A: #45892 — gather + 5D diagonal mask, via SDPA
# ---------------------------------------------------------------------------
def pr_45892_sdpa(q, compressed_kv, topk, scaling):
    """
    q:             [B, H, S, D]
    compressed_kv: [B, 1, T, D]
    topk:          [B, S, k]  (-1 marks invalid early-query slots)
    """
    B, H, S, D = q.shape
    T = compressed_kv.shape[2]
    k = topk.shape[-1]

    valid = topk >= 0
    safe_topk = topk.clamp(min=0)
    offsets = (torch.arange(B, device=compressed_kv.device) * T).view(B, 1, 1)
    flat_idx = (safe_topk + offsets).view(-1)
    flat_kv = compressed_kv.reshape(B * T, D)
    gathered = flat_kv.index_select(0, flat_idx).view(B, 1, S * k, D)

    block_bias = gathered.new_full((B, 1, S, S, k), float("-inf"))
    allowed = torch.where(valid, gathered.new_zeros(()), gathered.new_full((), float("-inf")))
    arange_s = torch.arange(S, device=gathered.device)
    block_bias[:, 0, arange_s, arange_s, :] = allowed
    block_bias = block_bias.view(B, 1, S, S * k)

    K = gathered.expand(B, H, S * k, D).contiguous()
    V = gathered.expand(B, H, S * k, D).contiguous()
    out = F.scaled_dot_product_attention(q, K, V, attn_mask=block_bias, scale=scaling)
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Path B: scatter-bias on un-gathered compressed_kv, via SDPA
# ---------------------------------------------------------------------------
def scatter_bias_sdpa(q, compressed_kv, topk, scaling):
    B, H, S, D = q.shape
    T = compressed_kv.shape[2]

    safe_topk = torch.where(topk >= 0, topk, torch.full_like(topk, T))
    bias = compressed_kv.new_full((B, 1, S, T + 1), float("-inf"))
    bias.scatter_(-1, safe_topk.unsqueeze(1), 0.0)
    bias = bias[..., :T]

    K = compressed_kv.expand(B, H, T, D).contiguous()
    V = compressed_kv.expand(B, H, T, D).contiguous()
    out = F.scaled_dot_product_attention(q, K, V, attn_mask=bias, scale=scaling)
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "case1": dict(B=2, S=8,    H=4, D=16, m=4, k=4),
    "case2": dict(B=1, S=256,  H=8, D=64, m=4, k=64),
    "4K":    dict(B=1, S=4096, H=8, D=64, m=4, k=128),
    "8K":    dict(B=1, S=8192, H=8, D=64, m=4, k=128),
}


def make_inputs(cfg, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, S, H, D, m, k = cfg["B"], cfg["S"], cfg["H"], cfg["D"], cfg["m"], cfg["k"]
    T = S // m
    q = torch.randn(B, H, S, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    compressed_kv = torch.randn(B, 1, T, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    topk = torch.full((B, S, k), -1, dtype=torch.long)
    for b in range(B):
        for i in range(S):
            threshold = (i + 1) // m
            if threshold <= 0:
                continue
            n_valid = min(k, threshold)
            perm = torch.randperm(threshold, generator=g)[:n_valid]
            topk[b, i, :n_valid] = perm
    topk = topk.to(device=device)
    scaling = 1.0 / math.sqrt(D)
    return q, compressed_kv, topk, scaling


# ---------------------------------------------------------------------------
# SDPA backend selection
# ---------------------------------------------------------------------------
def get_sdpa_ctx(name):
    """Return a context manager that pins SDPA to the requested backend."""
    if name == "auto" or not _HAS_SDPA_KERNEL:
        return nullcontext()
    backend_map = {
        "math":      [SDPBackend.MATH],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],  # fall back to math if mem-eff rejects
        "flash":     [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
    }
    return sdpa_kernel(backend_map[name])


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def cos_sim(a, b):
    return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def correctness(device, sdpa_ctx_name):
    print("=" * 78)
    print(f"CORRECTNESS  device={device}  sdpa_backend={sdpa_ctx_name}")
    print("=" * 78)
    for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
        for cfg_name, cfg in CONFIGS.items():
            pr_45892_attn_bytes = cfg["B"] * cfg["H"] * cfg["S"] * cfg["S"] * cfg["k"] * (2 if dtype != torch.float32 else 4)
            if device.type == "cuda" and pr_45892_attn_bytes > 6e9:
                print(f"  [skip] {cfg_name:>12s} | {dtype_name:<8s}  (#45892 attn tensor ~{pr_45892_attn_bytes/1e9:.1f} GB)")
                continue
            B, S, H, D, m, k = cfg["B"], cfg["S"], cfg["H"], cfg["D"], cfg["m"], cfg["k"]
            T = S // m
            print(f"\n--- {cfg_name} | {dtype_name}  | B={B} S={S} H={H} D={D} T={T} k={k} ---")
            print(f"   {'seed':>4} {'max_abs':>12} {'rel_max':>10} {'cos_sim':>10}")
            for seed in range(3):
                q, ckv, topk, scaling = make_inputs(cfg, dtype, device, seed=seed)
                with torch.no_grad(), get_sdpa_ctx(sdpa_ctx_name):
                    out_a = pr_45892_sdpa(q, ckv, topk, scaling)
                    out_b = scatter_bias_sdpa(q, ckv, topk, scaling)
                diff = (out_a.float() - out_b.float()).abs()
                ref = out_a.float().abs()
                max_abs = diff.max().item()
                rel = (diff / ref.clamp_min(1e-12)).max().item() * 100
                cs = cos_sim(out_a, out_b)
                print(f"   {seed:>4} {max_abs:>12.4e} {rel:>9.4f}% {cs:>10.6f}")


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_one(fn, args, device, iters, warmup, sdpa_ctx_name):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with torch.no_grad(), get_sdpa_ctx(sdpa_ctx_name):
            _ = fn(*args)
    sync(device)
    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad(), get_sdpa_ctx(sdpa_ctx_name):
            _ = fn(*args)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return median, peak


def benchmark(device, iters, warmup, sdpa_ctx_name):
    print("=" * 78)
    print(f"SPEED BENCHMARK  device={device}  iters={iters}  warmup={warmup}  sdpa_backend={sdpa_ctx_name}")
    print("=" * 78)
    dtypes = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]
    for dtype_name, dtype in dtypes:
        print(f"\n== dtype = {dtype_name} ==")
        header = f"  {'config':<14s} {'B/S/H/D':<16s} {'T/k':<10s} {'ratio':>8s} {'#45892 ms':>10s} {'ours ms':>10s} {'speedup':>9s} {'#45892 GB':>10s} {'ours GB':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cfg_name, cfg in CONFIGS.items():
            B, S, H, D, m, k = cfg["B"], cfg["S"], cfg["H"], cfg["D"], cfg["m"], cfg["k"]
            T = S // m
            ratio = (S * k) / T
            try:
                q, ckv, topk, scaling = make_inputs(cfg, dtype, device, seed=0)
            except Exception as e:
                print(f"  {cfg_name:<14s} setup error: {e}")
                continue

            def _run(fn):
                try:
                    ms, peak = bench_one(fn, (q, ckv, topk, scaling), device, iters, warmup, sdpa_ctx_name)
                    return f"{ms:>10.3f}", f"{peak/1e9:>10.3f}", ms
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    return f"{'OOM':>10s}", f"{'OOM':>10s}", float("inf")
                except Exception as e:
                    msg = type(e).__name__
                    return f"{msg:>10s}", f"{msg:>10s}", float("inf")

            a_str, a_gb, a_ms = _run(pr_45892_sdpa)
            b_str, b_gb, b_ms = _run(scatter_bias_sdpa)
            if a_ms != float("inf") and b_ms not in (0, float("inf")):
                sp = f"{a_ms/b_ms:>8.2f}x"
            else:
                sp = f"{'n/a':>9s}"
            print(f"  {cfg_name:<14s} {B}/{S}/{H}/{D:<10s} {T}/{k:<8s} {ratio:>7.1f}x {a_str} {b_str} {sp} {a_gb} {b_gb}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["correctness", "speed", "all"], default="all")
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--sdpa-backend", choices=["auto", "math", "efficient", "flash"], default="auto",
                   help="auto = let PyTorch pick; efficient = force mem-efficient (typical for custom masks)")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__}  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        cap = torch.cuda.get_device_capability(device)
        print(f"  compute capability: {cap[0]}.{cap[1]}")
    if not _HAS_SDPA_KERNEL and args.sdpa_backend != "auto":
        print(f"  WARNING: torch.nn.attention.sdpa_kernel unavailable; --sdpa-backend ignored")

    if args.mode in ("correctness", "all"):
        correctness(device, args.sdpa_backend)
    if args.mode in ("speed", "all"):
        benchmark(device, args.iters, args.warmup, args.sdpa_backend)


if __name__ == "__main__":
    main()
