"""
val_op_fused_recurrent_gated_delta_rule.py

python3 val_op_fused_recurrent_gated_delta_rule.py --device cuda --mode default
python3 val_op_fused_recurrent_gated_delta_rule.py --device cuda --mode reduce-overhead
python3 val_op_fused_recurrent_gated_delta_rule.py --device cuda --mode default --more

- observed pattern via warmup is ( cu_seqlens == arange(T+1) ) "each sequence length is 1",
  so token loop(recurrent) becomes 1-step update
- hence, reference is implemented as full vectorized -> remove torch.compile graph break
- and in case INPLACE_FINAL_STATE + continuous batching, state slot index is overlapped and raced,
  ssm_state_indices should be unique to do deterministic comparison in the validation situation
"""

import argparse
import os
import torch


# -----------------------------
# Vectorized Torch reference (observed warmup path)
# -----------------------------
@torch.no_grad()
def fused_recurrent_gated_delta_rule_torch_observed_vectorized(
    q: torch.Tensor,                      # (B=1, T, H=8, K=128) bf16
    k: torch.Tensor,                      # (1, T, 8, 128) bf16
    v: torch.Tensor,                      # (1, T, HV=16, V=128) bf16
    g: torch.Tensor,                      # (1, T, 16) fp32
    beta: torch.Tensor,                   # (1, T, 16) bf16
    scale: float,
    initial_state: torch.Tensor,          # (S, 16, 128, 128) fp32
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,           # (T+1,) int32 (must be arange)
    ssm_state_indices: torch.Tensor | None = None,    # (T,) int32 (must be unique for deterministic compare)
    num_accepted_tokens: torch.Tensor | None = None,  # None in observed
    use_qk_l2norm_in_kernel: bool = True,
):
    """
    Matches the exact branch combo seen in your logs:
      - IS_VARLEN=True with cu_seqlens = arange(T+1) -> each "sequence" length is 1
      - IS_CONTINUOUS_BATCHING=True with ssm_state_indices shape (T,)
      - INPLACE_FINAL_STATE=True
      - IS_KDA=False
      - IS_BETA_HEADWISE=False
      - USE_QK_L2NORM_IN_KERNEL=True
      - IS_SPEC_DECODING=False

    Returns:
      o: (1, T, HV, V) dtype = v.dtype
      final_state: same tensor as initial_state (inplace updates)
    """
    assert inplace_final_state, "Observed path uses INPLACE_FINAL_STATE=True."
    assert num_accepted_tokens is None, "Observed warmup shows IS_SPEC_DECODING=False."
    assert cu_seqlens is not None and ssm_state_indices is not None, "Observed path requires cu_seqlens + ssm_state_indices."
    assert q.shape[0] == 1 and k.shape[0] == 1 and v.shape[0] == 1, "Observed path B=1."

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    # Observed constants
    assert (H, HV, K, V) == (8, 16, 128, 128), f"Expected (H,HV,K,V)=(8,16,128,128), got {(H,HV,K,V)}"

    # cu_seqlens must be identity: [0,1,2,...,T]
    # (warmup logs show that exactly)
    # This implies each sequence has length 1 and bos == i_n.
    exp_cu = torch.arange(T + 1, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    if not torch.equal(cu_seqlens, exp_cu):
        raise AssertionError("This validator targets the observed warmup path: cu_seqlens must equal arange(T+1).")

    # Expand q/k from H=8 to HV=16 by repeating each head twice (group=2)
    group = HV // H
    assert group * H == HV
    # q_hv, k_hv: (1, T, HV, K)
    q_hv = q.repeat_interleave(group, dim=2)
    k_hv = k.repeat_interleave(group, dim=2)

    # Gather the state slots referenced by each token
    # idx: (T,) int64 for indexing
    idx = ssm_state_indices.to(torch.int64)  # (T,)
    if (idx < 0).any():
        raise AssertionError("This validator assumes all state indices are valid (>=0).")

    # H_batch: (T, HV, V, K) float32
    # (we drop batch dim B=1 for convenience)
    H_batch = initial_state.index_select(0, idx)  # gathers views/copies

    # Convert q,k,v,beta to float32 for math (kernel accumulates in fp32)
    # shapes:
    #   b_q: (T, HV, K)
    #   b_k: (T, HV, K)
    #   b_v: (T, HV, V)
    b_q = q_hv[0].to(torch.float32)
    b_k = k_hv[0].to(torch.float32)
    b_v = v[0].to(torch.float32)
    b_beta = beta[0].to(torch.float32).unsqueeze(-1)  # (T,HV,1)
    b_g = g[0].to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # (T,HV,1,1)

    if use_qk_l2norm_in_kernel:
        q_norm = torch.sqrt((b_q * b_q).sum(dim=-1, keepdim=True) + 1e-6)  # (T,HV,1)
        k_norm = torch.sqrt((b_k * b_k).sum(dim=-1, keepdim=True) + 1e-6)  # (T,HV,1)
        b_q = b_q / q_norm
        b_k = b_k / k_norm

    b_q = b_q * float(scale)

    # decay: H *= exp(g)
    H_batch = H_batch * torch.exp(b_g)

    # pred = H @ k  -> (T,HV,V)
    pred = torch.einsum("thvk,thk->thv", H_batch, b_k)
    b_v = (b_v - pred) * b_beta  # (T,HV,V)

    # H_new = H + v ⊗ k  -> (T,HV,V,K)
    H_new = H_batch + b_v.unsqueeze(-1) * b_k.unsqueeze(-2)

    # output o = H_new @ q  -> (T,HV,V)
    out = torch.einsum("thvk,thk->thv", H_new, b_q).to(v.dtype)  # bf16

    # Inplace final_state: scatter updated states back to initial_state
    # IMPORTANT: deterministic only if idx is unique
    initial_state.index_copy_(0, idx, H_new.to(initial_state.dtype))

    # Return with batch dim
    o = out.unsqueeze(0)  # (1,T,HV,V)
    return o, initial_state


def make_compiled(mode: str, scale: float, use_qk_l2norm_in_kernel: bool):
    # capture constants for stability
    def fn(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices):
        return fused_recurrent_gated_delta_rule_torch_observed_vectorized(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    return torch.compile(fn, mode=mode, fullgraph=False)


# -----------------------------
# error metrics helpers
# -----------------------------
def max_abs_err(x, y):
    return (x - y).abs().max().item()


def max_rel_err(x, y, eps=1e-8):
    denom = y.abs().clamp_min(eps)
    return ((x - y).abs() / denom).max().item()


def compare(name, x, y, atol, rtol):
    abs_err = max_abs_err(x, y)
    rel_err = max_rel_err(x, y)
    ok = (abs_err <= atol) or (rel_err <= rtol)
    print(f"{name:>18} | abs_max={abs_err:.3e} rel_max={rel_err:.3e} | "
          f"atol={atol:.3e} rtol={rtol:.3e} | {'OK' if ok else 'FAIL'}")
    return ok


def run_case(
    triton_fn,
    compiled_fn,
    *,
    T: int,
    S: int,
    device,
    dtype_qkv,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    seed: int,
    atol: float,
    rtol: float,
):
    torch.manual_seed(seed)

    # Observed constants
    B, H, HV, K, V = 1, 8, 16, 128, 128
    assert T <= S, "Need S >= T to generate unique state indices."

    # Inputs
    q = torch.randn((B, T, H, K), device=device, dtype=dtype_qkv)
    k = torch.randn((B, T, H, K), device=device, dtype=dtype_qkv)
    v = torch.randn((B, T, HV, V), device=device, dtype=dtype_qkv)

    g = torch.randn((B, T, HV), device=device, dtype=torch.float32)
    beta = torch.randn((B, T, HV), device=device, dtype=dtype_qkv)

    # Warmup-identity varlen pattern
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.int32)

    # IMPORTANT: UNIQUE indices to avoid race/nondeterminism in INPLACE_FINAL_STATE path
    # (this is why your previous run failed only for large T)
    ssm_state_indices = torch.randperm(S, device=device)[:T].to(torch.int32)

    # initial_state: fp32
    initial_state0 = torch.randn((S, HV, V, K), device=device, dtype=torch.float32)

    # Clone because Triton path updates inplace
    initial_state_triton = initial_state0.clone()
    initial_state_comp = initial_state0.clone()

    # Run Triton
    o_tri, st_tri = triton_fn(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state_triton,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Run compiled torch reference
    o_cmp, st_cmp = compiled_fn(q, k, v, g, beta, initial_state_comp, cu_seqlens, ssm_state_indices)

    # Checks (match vLLM wrapper return: o shape (B,T,HV,V))
    assert o_tri.shape == (B, T, HV, V), f"Unexpected o_tri shape: {tuple(o_tri.shape)}"
    assert o_cmp.shape == (B, T, HV, V), f"Unexpected o_cmp shape: {tuple(o_cmp.shape)}"
    assert o_tri.dtype == dtype_qkv and o_cmp.dtype == dtype_qkv, "Output dtype mismatch"
    assert st_tri.shape == (S, HV, V, K) and st_cmp.shape == (S, HV, V, K), "State shape mismatch"
    assert st_tri.dtype == torch.float32 and st_cmp.dtype == torch.float32, "State dtype mismatch"

    ok_o = compare("o (output)", o_tri, o_cmp, atol, rtol)

    # Compare touched slots only (should be exactly T unique)
    touched = ssm_state_indices.to(torch.int64)
    st_tri_t = st_tri.index_select(0, touched)
    st_cmp_t = st_cmp.index_select(0, touched)
    ok_s = compare("state (touched)", st_tri_t, st_cmp_t, atol * 5, rtol * 5)

    ok = ok_o and ok_s
    print(f"CASE T={T} S={S} dtype={dtype_qkv} l2norm={use_qk_l2norm_in_kernel} scale={scale}: {'PASS' if ok else 'FAIL'}")
    print("-" * 100)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--scale", type=float, default=0.08838834764831845)
    ap.add_argument("--slots", type=int, default=512,
                    help="Number of state slots S (logs show 3624; smaller is fine). Must satisfy S >= max(T).")
    ap.add_argument("--more", action="store_true",
                    help="Run more T values (matches warmup-style sweep).")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(args.device)
    dtype_qkv = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    # tolerances
    atol, rtol = (2e-2, 2e-2)

    # Import vLLM Triton wrapper
    try:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd as triton_op
    except Exception as e:
        raise RuntimeError(
            "Failed to import vLLM fused_recurrent_gated_delta_rule_fwd from "
            "vllm.model_executor.layers.fla.ops.fused_recurrent.\n"
            "Make sure you're running in the vLLM environment.\n"
            f"Original error: {e}"
        )

    def triton_fn(
        q, k, v, g, beta, scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    ):
        return triton_op(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    compiled_fn = make_compiled(mode=args.mode, scale=args.scale, use_qk_l2norm_in_kernel=True)

    # Warmup compile (small T)
    print(f"PID={os.getpid()} Warming up torch.compile...")
    B, H, HV, K, V = 1, 8, 16, 128, 128
    T_warm = 8
    q0 = torch.randn((B, T_warm, H, K), device=device, dtype=dtype_qkv)
    k0 = torch.randn((B, T_warm, H, K), device=device, dtype=dtype_qkv)
    v0 = torch.randn((B, T_warm, HV, V), device=device, dtype=dtype_qkv)
    g0 = torch.randn((B, T_warm, HV), device=device, dtype=torch.float32)
    beta0 = torch.randn((B, T_warm, HV), device=device, dtype=dtype_qkv)
    cu0 = torch.arange(T_warm + 1, device=device, dtype=torch.int32)
    idx0 = torch.randperm(args.slots, device=device)[:T_warm].to(torch.int32)
    st0 = torch.randn((args.slots, HV, V, K), device=device, dtype=torch.float32)

    _ = compiled_fn(q0, k0, v0, g0, beta0, st0, cu0, idx0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup done.\n")

    Ts = [256, 128, 32, 1]
    if args.more:
        Ts = [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128,
              120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]

    if max(Ts) > args.slots:
        raise RuntimeError(f"--slots must be >= max(T). Got slots={args.slots}, max(T)={max(Ts)}")

    all_ok = True
    for i, T in enumerate(Ts):
        ok = run_case(
            triton_fn=triton_fn,
            compiled_fn=compiled_fn,
            T=T,
            S=args.slots,
            device=device,
            dtype_qkv=dtype_qkv,
            scale=args.scale,
            use_qk_l2norm_in_kernel=True,
            seed=2026 + i * 17,
            atol=atol,
            rtol=rtol,
        )
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
