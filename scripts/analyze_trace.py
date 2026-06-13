"""Quantify a torch.profiler chrome trace: GPU-busy %, idle gaps, host<->device
syncs, H2D/D2H copies, and the top kernels.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
ITEM_NAMES = ("aten::item", "aten::_local_scalar_dense")


def _merge_busy(intervals):
    """Total covered time of a set of [start,end] intervals (union)."""
    if not intervals:
        return 0.0, 0.0, 0.0
    intervals.sort()
    start = intervals[0][0]
    end = intervals[-1][1]
    busy = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    return busy, start, end


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace", nargs="?", default=None)
    p.add_argument("--out_dir", default="./out_profile")
    p.add_argument("--gap_us", type=float, default=30.0, help="idle-gap threshold (us)")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    path = args.trace
    if path is None:
        cands = sorted(
            glob.glob(os.path.join(args.out_dir, "trace_*.json")), key=os.path.getmtime
        )
        if not cands:
            raise SystemExit(f"no trace_*.json in {args.out_dir}")
        path = cands[-1]
    print(f"== analyzing {path} ==\n")

    with open(path) as f:
        events = json.load(f).get("traceEvents", [])

    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]

    gpu_by_stream = defaultdict(list)
    for e in complete:
        if e.get("cat") in GPU_CATS:
            s = e["ts"]
            gpu_by_stream[e["tid"]].append((s, s + e["dur"], e.get("name", "")))

    all_gpu = [(s, en) for ivs in gpu_by_stream.values() for (s, en, _) in ivs]
    busy, gstart, gend = _merge_busy([(s, en) for (s, en) in all_gpu])
    span = gend - gstart if gend > gstart else 1.0
    print(
        f"GPU union busy: {busy/1000:.2f} ms / {span/1000:.2f} ms wall = "
        f"{100*busy/span:.1f}%   (idle {100*(1-busy/span):.1f}%)"
    )

    stream_time = {
        tid: sum(en - s for (s, en, _) in ivs) for tid, ivs in gpu_by_stream.items()
    }
    busy_tid = max(stream_time, key=stream_time.get)
    ivs = sorted(gpu_by_stream[busy_tid])
    sbusy, ss, se = _merge_busy([(s, en) for (s, en, _) in ivs])
    sspan = se - ss if se > ss else 1.0
    print(
        f"compute stream tid={busy_tid}: busy {100*sbusy/sspan:.1f}%  "
        f"({len(ivs)} gpu events)\n"
    )

    gaps = []
    for (s0, e0, n0), (s1, e1, n1) in zip(ivs, ivs[1:]):
        g = s1 - e0
        if g >= args.gap_us:
            gaps.append((g, n0, n1))
    gaps.sort(reverse=True)
    total_gap = sum(g for g, _, _ in gaps)
    print(
        f"idle gaps >= {args.gap_us:.0f}us on compute stream: {len(gaps)} gaps, "
        f"{total_gap/1000:.2f} ms total"
    )
    for g, n0, n1 in gaps[:12]:
        print(f"    {g:8.1f} us   after [{n0[:42]}]  before [{n1[:42]}]")
    print()

    runtime = [e for e in complete if e.get("cat") == "cuda_runtime"]
    n_sync = sum(1 for e in runtime if any(k in e.get("name", "") for k in SYNC_NAMES))
    n_item = sum(1 for e in complete if e.get("name", "") in ITEM_NAMES)
    print(f"host<->device sync points:")
    print(f"    cudaStream/Device/EventSynchronize calls : {n_sync}")
    print(f"    aten::item / _local_scalar_dense (D2H)   : {n_item}")
    print(
        "    (legit baseline: ~1 .item() per log window; per-step syncs = a finding)\n"
    )

    memcpy = defaultdict(lambda: [0, 0.0, 0])
    for e in complete:
        if e.get("cat") == "gpu_memcpy":
            rec = memcpy[e.get("name", "memcpy")]
            rec[0] += 1
            rec[1] += e["dur"]
            rec[2] += int(e.get("args", {}).get("bytes", 0) or 0)
    if memcpy:
        print("GPU memcpy:")
        for name, (c, dur, b) in sorted(memcpy.items(), key=lambda x: -x[1][1]):
            print(
                f"    {name[:40]:40s} x{c:<5d} {dur/1000:7.2f} ms  {b/2**20:8.2f} MiB"
            )
        print()

    kern = defaultdict(lambda: [0, 0.0])
    for e in complete:
        if e.get("cat") == "kernel":
            rec = kern[e.get("name", "")]
            rec[0] += 1
            rec[1] += e["dur"]
    print(f"top {args.top} GPU kernels by total time:")
    tot = sum(d for _, d in kern.values()) or 1.0
    for name, (c, dur) in sorted(kern.items(), key=lambda x: -x[1][1])[: args.top]:
        print(f"    {100*dur/tot:5.1f}%  {dur/1000:8.2f} ms  x{c:<6d} {name[:74]}")


if __name__ == "__main__":
    main()
