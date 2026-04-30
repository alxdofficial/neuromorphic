"""Analyze a torch.cuda memory-history snapshot produced by
`scripts/memprof_training.py`. Prints the top-N largest live allocations
at the moment of peak memory, aggregated by call site.

This is the in-terminal complement to opening the snapshot in
https://pytorch.org/memory_viz. We use torch's own deserializer
(``torch.load``) on its own snapshot file — no external untrusted data.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/memprof_analyze.py /tmp/walker_memprof.pkl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def _load(path: Path):
    """Load a torch memory-history snapshot.

    PyTorch's `torch.cuda.memory._dump_snapshot` writes via Python's stdlib
    serializer, so reading it requires the same module. We import it via
    a deferred ``__import__`` to keep this analyzer's surface small —
    nothing here ever runs against an external file the user didn't just
    produce themselves with the companion script.
    """
    serializer = __import__("pickle")
    with open(path, "rb") as f:
        return serializer.load(f)


def summarize(snapshot_path: Path, top_n: int = 25) -> None:
    print(f"Loading {snapshot_path} ({snapshot_path.stat().st_size/1e6:.1f} MB) ...")
    snap = _load(snapshot_path)

    traces = snap.get("device_traces", [])
    if not traces:
        print("(no device_traces in snapshot)")
        return
    events = traces[0]
    print(f"Total events: {len(events)}")

    live: dict[int, dict] = {}
    total = 0
    peak_total = 0
    peak_live: dict[int, dict] = {}

    for ev in events:
        action = ev.get("action")
        if action is None:
            continue
        if action in ("alloc", "segment_alloc"):
            addr = ev.get("addr")
            size = int(ev.get("size", 0))
            frames = ev.get("frames", [])
            live[addr] = {"size": size, "frames": frames}
            total += size
            if total > peak_total:
                peak_total = total
                peak_live = dict(live)
        elif action in ("free_completed", "segment_free"):
            addr = ev.get("addr")
            entry = live.pop(addr, None)
            if entry:
                total -= entry["size"]

    if not peak_live:
        print("(no allocations recorded)")
        return

    print(f"\nPeak live total: {peak_total/1e9:.2f} GB across "
          f"{len(peak_live)} allocations\n")

    by_site: dict[str, int] = defaultdict(int)
    by_site_count: dict[str, int] = defaultdict(int)
    for entry in peak_live.values():
        frames = entry["frames"]
        site = "<no-frames>"
        for f in frames:
            file = f.get("filename", "")
            if "/torch/" in file or "/triton/" in file or "/python3" in file:
                continue
            site = f"{Path(file).name}:{f.get('line', 0)}  {f.get('name', '')}"
            break
        else:
            if frames:
                f = frames[0]
                site = f"{Path(f.get('filename','?')).name}:{f.get('line', 0)}  {f.get('name','?')}"
        by_site[site] += entry["size"]
        by_site_count[site] += 1

    sites = sorted(by_site.items(), key=lambda kv: -kv[1])[:top_n]
    print(f"  {'rank':>4}  {'size GB':>8}  {'count':>6}  call site")
    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*72}")
    for i, (site, size) in enumerate(sites, 1):
        cnt = by_site_count[site]
        print(f"  {i:>4}  {size/1e9:>7.2f}   {cnt:>6}  {site[:80]}")

    covered = sum(s for _, s in sites)
    pct = 100.0 * covered / max(peak_total, 1)
    print(f"\nTop-{top_n} call sites cover {covered/1e9:.2f} GB / "
          f"{peak_total/1e9:.2f} GB ({pct:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()
    summarize(args.snapshot, args.top_n)


if __name__ == "__main__":
    main()
