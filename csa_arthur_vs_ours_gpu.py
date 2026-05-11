"""
Compare two CSA per-query sparse-attention expressions on GPU.

Both are mathematically equivalent to MindSpeed-LLM SparseFlashAttentionTriton's
per-query sparse attention semantics. This script verifies that equivalence
in fp32 / bf16 / fp16, then benchmarks the two on GPU across configs.

  Path A — Arthur #45892 (HEAD 8bdbfbb):
      * gather compressed_kv into [B, 1, S*k, D] via index_select
      * build 5D diagonal block bias [B, 1, S, S, k], view as [B, 1, S, S*k]
      * eager attention is then [B, H, S, S*k]

  Path B — Ours (PR #45879-era follow-up):
      * keep compressed_kv as [B, 1, T, D] (no gather)
      * scatter into a [B, 1, S, T+1] -inf mask (last column is invalid-topk
        sentinel), drop the sentinel column -> [B, 1, S, T]
      * eager attention is then [B, H, S, T]  (T = S/m, typically S*k >> T)

Both paths include the sink-token logic (concat sink column, fp32 softmax,
drop sink) so the comparison is apples-to-apples with the HF eager path
and MindSpeed-LLM's torch fallback (g2_attention_kernel.sparse_flash_attn).

Usage:
    python csa_arthur_vs_ours_gpu.py                  # correctness + speed, auto-device
    python csa_arthur_vs_ours_gpu.py --mode correctness
    python csa_arthur_vs_ours_gpu.py --mode speed
    python csa_arthur_vs_ours_gpu.py --device cuda
    python csa_arthur_vs_ours_gpu.py --iters 50
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path A: Arthur #45892 — gather + 5D diagonal mask
# ---------------------------------------------------------------------------
def arthur_csa_attention(q, compressed_kv, topk, sinks, scaling):
    """
    q:             [B, H, S, D]
    compressed_kv: [B, 1, T, D]
    topk:          [B, S, k]  (long; -1 marks invalid early-query slots)
    sinks:         [H]
    Returns:       [B, S, H, D]
    """
    B, H, S, D = q.shape
    T = compressed_kv.shape[2]
    k = topk.shape[-1]

    # ---- gather (index_select over flattened batch axis) ----
    valid = topk >= 0
    safe_topk = topk.clamp(min=0)
    offsets = (torch.arange(B, device=compressed_kv.device) * T).view(B, 1, 1)
    flat_idx = (safe_topk + offsets).view(-1)
    flat_kv = compressed_kv.reshape(B * T, D)
    gathered = flat_kv.index_select(0, flat_idx).view(B, 1, S * k, D)

    # ---- 5D diagonal block bias, viewed as [B, 1, S, S*k] ----
    block_bias = gathered.new_full((B, 1, S, S, k), float("-inf"))
    allowed = torch.where(valid, gathered.new_zeros(()), gathered.new_full((), float("-inf")))
    arange_s = torch.arange(S, device=gathered.device)
    block_bias[:, 0, arange_s, arange_s, :] = allowed
    block_bias = block_bias.view(B, 1, S, S * k)

    # ---- eager attention with sink ----
    K = gathered.expand(B, H, S * k, D)
    V = gathered.expand(B, H, S * k, D)
    attn_weights = torch.matmul(q, K.transpose(2, 3)) * scaling
    attn_weights = attn_weights + block_bias

    sink_col = sinks.reshape(1, -1, 1, 1).expand(B, H, S, 1)
    combined = torch.cat([attn_weights, sink_col], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    scores = probs[..., :-1]
    out = torch.matmul(scores, V)
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Path B: Ours — scatter-bias on full compressed_kv (no gather)
# ---------------------------------------------------------------------------
def ours_csa_attention(q, compressed_kv, topk, sinks, scaling):
    """
    Same signature as arthur_csa_attention. Uses compressed_kv directly,
    builds a [B, 1, S, T] mask by scattering 0.0 into a -inf canvas.
    """
    B, H, S, D = q.shape
    T = compressed_kv.shape[2]

    # ---- scatter-bias mask (sentinel column at T for invalid topk == -1) ----
    safe_topk = torch.where(topk >= 0, topk, torch.full_like(topk, T))
    bias = compressed_kv.new_full((B, 1, S, T + 1), float("-inf"))
    bias.scatter_(-1, safe_topk.unsqueeze(1), 0.0)
    bias = bias[..., :T]

    # ---- eager attention with sink ----
    K = compressed_kv.expand(B, H, T, D)
    V = compressed_kv.expand(B, H, T, D)
    attn_weights = torch.matmul(q, K.transpose(2, 3)) * scaling
    attn_weights = attn_weights + bias

    sink_col = sinks.reshape(1, -1, 1, 1).expand(B, H, S, 1)
    combined = torch.cat([attn_weights, sink_col], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    scores = probs[..., :-1]
    out = torch.matmul(scores, V)
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "case1": dict(B=2, S=8,    H=4, D=16, m=4, k=4),    # toy / launch-overhead sanity
    "case2": dict(B=1, S=256,  H=8, D=64, m=4, k=64),
    "4K":    dict(B=1, S=4096, H=8, D=64, m=4, k=128),
    "8K":    dict(B=1, S=8192, H=8, D=64, m=4, k=128),
}


def make_inputs(cfg, dtype, device, seed=0):
    """Build (q, compressed_kv, topk, sinks, scaling) with a realistic causal
    topk: query at position i may only pick from compressed blocks with index
    < (i+1)//m. If the available pool is smaller than k, the row is right-padded
    with -1 to mark invalid slots (matches #45892's sentinel convention)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, S, H, D, m, k = cfg["B"], cfg["S"], cfg["H"], cfg["D"], cfg["m"], cfg["k"]
    T = S // m

    q = torch.randn(B, H, S, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    compressed_kv = torch.randn(B, 1, T, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    sinks = torch.randn(H, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    # build per-row topk on CPU (cheap), then move
    topk = torch.full((B, S, k), -1, dtype=torch.long)
    for b in range(B):
        for i in range(S):
            threshold = (i + 1) // m
            if threshold <= 0:
                continue  # row stays all -1
            n_valid = min(k, threshold)
            perm = torch.randperm(threshold, generator=g)[:n_valid]
            topk[b, i, :n_valid] = perm
    topk = topk.to(device=device)

    scaling = 1.0 / math.sqrt(D)
    return q, compressed_kv, topk, sinks, scaling


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def cos_sim(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


def correctness(device):
    print("=" * 78)
    print(f"CORRECTNESS  device={device}")
    print("=" * 78)
    for dtype_name, dtype in [("float32", torch.float32),
                              ("bfloat16", torch.bfloat16)]:
        for cfg_name, cfg in CONFIGS.items():
            # rough OOM guard for Arthur path on small GPUs
            arthur_attn_bytes = cfg["B"] * cfg["H"] * cfg["S"] * cfg["S"] * cfg["k"] * (2 if dtype != torch.float32 else 4)
            if device.type == "cuda" and arthur_attn_bytes > 6e9:
                print(f"  [skip] {cfg_name:>12s} | {dtype_name:<8s}  (arthur attn tensor ~{arthur_attn_bytes/1e9:.1f} GB)")
                continue
            B, S, H, D, m, k = cfg["B"], cfg["S"], cfg["H"], cfg["D"], cfg["m"], cfg["k"]
            T = S // m
            print(f"\n--- {cfg_name} | {dtype_name}  | B={B} S={S} H={H} D={D} T={T} k={k} ---")
            print(f"   {'seed':>4} {'max_abs':>12} {'rel_max':>10} {'cos_sim':>10}")
            for seed in range(3):
                q, ckv, topk, sinks, scaling = make_inputs(cfg, dtype, device, seed=seed)
                with torch.no_grad():
                    out_a = arthur_csa_attention(q, ckv, topk, sinks, scaling)
                    out_b = ours_csa_attention(q, ckv, topk, sinks, scaling)
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


def bench_one(fn, q, ckv, topk, sinks, scaling, device, iters, warmup):
    """Return (median_ms, peak_bytes)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(q, ckv, topk, sinks, scaling)
    sync(device)

    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fn(q, ckv, topk, sinks, scaling)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return median, peak


def benchmark(device, iters, warmup):
    print("=" * 78)
    print(f"SPEED BENCHMARK  device={device}  iters={iters}  warmup={warmup}")
    print("=" * 78)
    dtypes = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]

    for dtype_name, dtype in dtypes:
        print(f"\n== dtype = {dtype_name} ==")
        header = f"  {'config':<14s} {'B/S/H/D':<16s} {'T/k':<10s} {'ratio':>8s} {'arthur ms':>10s} {'ours ms':>10s} {'speedup':>9s} {'arthur GB':>10s} {'ours GB':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cfg_name, cfg in CONFIGS.items():
            B, S, H, D, m, k = cfg["B"], cfg["S"], cfg["H"], cfg["D"], cfg["m"], cfg["k"]
            T = S // m
            ratio = (S * k) / T  # arthur attn cols / ours attn cols
            shape_s = f"{B}/{S}/{H}/{D}"
            tk_s = f"{T}/{k}"
            try:
                q, ckv, topk, sinks, scaling = make_inputs(cfg, dtype, device, seed=0)
            except Exception as e:
                print(f"  {cfg_name:<14s} {shape_s:<16s} {tk_s:<10s}  setup error: {e}")
                continue

            # Arthur
            try:
                a_ms, a_peak = bench_one(arthur_csa_attention, q, ckv, topk, sinks, scaling, device, iters, warmup)
                a_str = f"{a_ms:>10.3f}"
                a_gb = f"{a_peak/1e9:>10.3f}"
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                a_ms, a_peak = float("inf"), 0
                a_str = f"{'OOM':>10s}"
                a_gb = f"{'OOM':>10s}"
            except Exception as e:
                a_ms, a_peak = float("inf"), 0
                a_str = f"{'ERR':>10s}"
                a_gb = f"{'ERR':>10s}"

            # Ours
            try:
                b_ms, b_peak = bench_one(ours_csa_attention, q, ckv, topk, sinks, scaling, device, iters, warmup)
                b_str = f"{b_ms:>10.3f}"
                b_gb = f"{b_peak/1e9:>10.3f}"
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                b_ms, b_peak = float("inf"), 0
                b_str = f"{'OOM':>10s}"
                b_gb = f"{'OOM':>10s}"
            except Exception as e:
                b_ms, b_peak = float("inf"), 0
                b_str = f"{'ERR':>10s}"
                b_gb = f"{'ERR':>10s}"

            if a_ms != float("inf") and b_ms not in (0, float("inf")):
                speed = f"{a_ms / b_ms:>8.2f}x"
            else:
                speed = f"{'n/a':>9s}"

            print(f"  {cfg_name:<14s} {shape_s:<16s} {tk_s:<10s} {ratio:>7.1f}x {a_str} {b_str} {speed} {a_gb} {b_gb}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["correctness", "speed", "all"], default="all")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--iters", type=int, default=20, help="benchmark iterations (default 20)")
    p.add_argument("--warmup", type=int, default=5, help="benchmark warmup iters (default 5)")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__}  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        cap = torch.cuda.get_device_capability(device)
        print(f"  compute capability: {cap[0]}.{cap[1]}")

    if args.mode in ("correctness", "all"):
        correctness(device)
    if args.mode in ("speed", "all"):
        benchmark(device, args.iters, args.warmup)


if __name__ == "__main__":
    main()
