#!/usr/bin/env python3
"""Quick test: Triton kernel on non-default CUDA stream + event sync."""
import torch
import triton
import triton.language as tl


@triton.jit
def _test_kernel(out_ptr, val, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.full([N], val, dtype=tl.float32))


def main():
    device = torch.device('cuda:0')
    main_stream = torch.cuda.current_stream(device)
    eb_stream = torch.cuda.Stream(device)

    buf_main = torch.zeros(256, device=device, dtype=torch.float32)
    buf_eb = torch.zeros(256, device=device, dtype=torch.float32)

    # Launch on main stream
    _test_kernel[(1,)](buf_main, 1.0, N=256)

    # Launch on eb_stream
    with torch.cuda.stream(eb_stream):
        _test_kernel[(1,)](buf_eb, 2.0, N=256)

    torch.cuda.synchronize()
    ok1 = buf_main[0].item() == 1.0 and buf_eb[0].item() == 2.0
    print(f"Test 1 — Triton on non-default stream: {'PASS' if ok1 else 'FAIL'}")
    print(f"  main: {buf_main[:4].tolist()}, eb: {buf_eb[:4].tolist()}")

    # Test event-based cross-stream sync
    event = torch.cuda.Event()
    src = torch.zeros(256, device=device)
    dst = torch.zeros(256, device=device)

    src.fill_(42.0)
    event.record(main_stream)

    with torch.cuda.stream(eb_stream):
        eb_stream.wait_event(event)
        _test_kernel[(1,)](dst, 99.0, N=256)

    torch.cuda.synchronize()
    ok2 = dst[0].item() == 99.0
    print(f"Test 2 — Event-based cross-stream sync: {'PASS' if ok2 else 'FAIL'}")
    print(f"  dst: {dst[:4].tolist()}")

    # Test: eb_stream reads tensor written on main stream
    data = torch.zeros(256, device=device)
    result = torch.zeros(256, device=device)

    data.fill_(7.0)  # main stream writes
    event2 = torch.cuda.Event()
    event2.record(main_stream)

    with torch.cuda.stream(eb_stream):
        eb_stream.wait_event(event2)
        # Read data (written on main) and write result (on eb)
        result.copy_(data)

    torch.cuda.synchronize()
    ok3 = result[0].item() == 7.0
    print(f"Test 3 — Cross-stream tensor read: {'PASS' if ok3 else 'FAIL'}")
    print(f"  result: {result[:4].tolist()}")

    if ok1 and ok2 and ok3:
        print("\nAll tests PASS. CUDA stream overlap is feasible.")
    else:
        print("\nSome tests FAILED.")


if __name__ == "__main__":
    main()
