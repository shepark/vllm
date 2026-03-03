"""
python3 val_op_fused_gdn_gating.py --device cuda --dtype bf16 --B 2048 --H 16
python3 val_op_fused_gdn_gating.py --device cuda --dtype bf16 --B 2048 --H 16 --more
python3 val_op_fused_gdn_gating.py --device cuda --dtype fp16 --B 2048 --H 16 --more
python3 val_op_fused_gdn_gating.py --device cuda --dtype fp32 --B 2048 --H 16 --more
"""
import argparse
import torch
import torch.nn.functional as F


def fused_gdn_gating_torch(A_log, a, b, dt_bias, beta: float = 1.0, threshold: float = 20.0):
    """
    Match vLLM fused_gdn_gating() behavior observed in runtime logs:
      - A_log: (H,)          (H == num_heads, e.g., 16)
      - dt_bias: (H,)
      - a, b: (B, H)         (B == token batch, e.g., 2048 / 1448)
      - g: float32, shape (1, B, H)
      - beta_output: dtype of b, shape (1, B, H)
    """
    x = a.to(torch.float32) + dt_bias.to(torch.float32)  # (B, H)
    sp = F.softplus(x, beta=beta, threshold=threshold)   # (B, H)
    g = -torch.exp(A_log.to(torch.float32)).unsqueeze(0) * sp.unsqueeze(0)  # (1, B, H) float32
    beta_output = torch.sigmoid(b.to(torch.float32)).to(b.dtype).unsqueeze(0)  # (1, B, H) b.dtype
    return g, beta_output


def make_compiled(beta: float, threshold: float, mode: str):
    # Capture beta/threshold as constants in the graph for better fusion stability.
    def fn(A_log, a, b, dt_bias):
        return fused_gdn_gating_torch(A_log, a, b, dt_bias, beta=beta, threshold=threshold)

    return torch.compile(fn, mode=mode, fullgraph=False)


def max_abs_err(x, y):
    return (x - y).abs().max().item()


def max_rel_err(x, y, eps=1e-8):
    denom = y.abs().clamp_min(eps)
    return ((x - y).abs() / denom).max().item()


def compare(name, x, y, atol, rtol):
    abs_err = max_abs_err(x, y)
    rel_err = max_rel_err(x, y)
    ok = (abs_err <= atol) or (rel_err <= rtol)
    print(f"{name:>14} | abs_max={abs_err:.3e} rel_max={rel_err:.3e} | "
          f"atol={atol:.3e} rtol={rtol:.3e} | {'OK' if ok else 'FAIL'}")
    return ok


def run_case(triton_fn, compiled_fn, B, H, dtype, device, beta, threshold, seed, atol, rtol):
    torch.manual_seed(seed)

    # IMPORTANT: match vLLM signature
    A_log = torch.randn(H, device=device, dtype=dtype)       # (H,)
    dt_bias = torch.randn(H, device=device, dtype=dtype)     # (H,)
    a = torch.randn(B, H, device=device, dtype=dtype)        # (B, H)
    b = torch.randn(B, H, device=device, dtype=dtype)        # (B, H)

    g_tri, beta_tri = triton_fn(A_log, a, b, dt_bias, beta=beta, threshold=threshold)
    g_cmp, beta_cmp = compiled_fn(A_log, a, b, dt_bias)

    # Sanity checks based on vLLM behavior
    assert g_tri.shape == (1, B, H), f"Unexpected g_tri shape: {tuple(g_tri.shape)}"
    assert beta_tri.shape == (1, B, H), f"Unexpected beta_tri shape: {tuple(beta_tri.shape)}"
    assert g_tri.dtype == torch.float32, f"Expected g_tri dtype float32, got {g_tri.dtype}"
    assert beta_tri.dtype == dtype, f"Expected beta_tri dtype {dtype}, got {beta_tri.dtype}"

    assert g_cmp.shape == (1, B, H), f"Unexpected g_cmp shape: {tuple(g_cmp.shape)}"
    assert beta_cmp.shape == (1, B, H), f"Unexpected beta_cmp shape: {tuple(beta_cmp.shape)}"
    assert g_cmp.dtype == torch.float32, f"Expected g_cmp dtype float32, got {g_cmp.dtype}"
    assert beta_cmp.dtype == dtype, f"Expected beta_cmp dtype {dtype}, got {beta_cmp.dtype}"

    ok1 = compare("g_forward", g_tri, g_cmp, atol, rtol)
    ok2 = compare("beta_forward", beta_tri, beta_cmp, atol, rtol)
    print(f"CASE B={B} H={H} dtype={dtype} beta={beta} threshold={threshold}: {'PASS' if (ok1 and ok2) else 'FAIL'}")
    print("-" * 90)
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=20.0)

    # Defaults updated to match your observed runtime logs:
    #   H == num_heads == 16
    #   B == token batch sizes (often 2048, sometimes 1448)
    ap.add_argument("--B", type=int, default=2048)
    ap.add_argument("--H", type=int, default=16)

    ap.add_argument("--more", action="store_true",
                    help="Run additional shapes including the observed (1448, H) case.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # tolerance
    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-6, 1e-6

    # Import vLLM Triton op
    try:
        from vllm.model_executor.models.qwen3_next import fused_gdn_gating as fused_gdn_gating_triton
    except Exception as e:
        raise RuntimeError(
            "Failed to import vLLM Triton fused_gdn_gating from vllm.model_executor.models.qwen3_next.\n"
            "Make sure you're running in the same environment as vLLM, and any Triton-related import issues are fixed.\n"
            f"Original error: {e}"
        )

    def triton_fn(A_log, a, b, dt_bias, beta, threshold):
        return fused_gdn_gating_triton(A_log, a, b, dt_bias, beta=beta, threshold=threshold)

    compiled_fn = make_compiled(beta=args.beta, threshold=args.threshold, mode=args.mode)

    # warmup compile
    print("Warming up compilation...")
    _ = compiled_fn(
        torch.randn(args.H, device=device, dtype=dtype),
        torch.randn(args.B, args.H, device=device, dtype=dtype),
        torch.randn(args.B, args.H, device=device, dtype=dtype),
        torch.randn(args.H, device=device, dtype=dtype),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup done.\n")

    cases = [(args.B, args.H)]
    if args.more:
        # Add the runtime-observed batch size + a couple minor variants
        cases += [
            (1448, args.H),
            (256, args.H),
            (4096, args.H),
        ]

    all_ok = True
    for i, (B, H) in enumerate(cases):
        all_ok = run_case(
            triton_fn=triton_fn,
            compiled_fn=compiled_fn,
            B=B,
            H=H,
            dtype=dtype,
            device=device,
            beta=args.beta,
            threshold=args.threshold,
            seed=1234 + i * 17,
            atol=atol,
            rtol=rtol,
        ) and all_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    # stricter numerics
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
