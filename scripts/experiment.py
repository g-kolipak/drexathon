"""Step 3: command-line accuracy test on a 2x5 grid, no UI.

Usage: uv run scripts/experiment.py --file data/sample.txt --rows 2 --cols 5
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unshred.matcher import (  # noqa: E402
    match,
    neighbor_accuracy,
    neighbor_stats,
)
from unshred.shredder import shred  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/sample.txt")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="drex-v2.0")
    args = ap.parse_args()

    text = Path(args.file).read_text()
    shr = shred(text, rows=args.rows, cols=args.cols, seed=args.seed)
    print(f"Shredded {len(text)} chars into {args.rows}x{args.cols} "
          f"= {len(shr.pieces)} pieces")

    t0 = time.monotonic()
    result = match(shr, model=args.model)
    wall = time.monotonic() - t0

    correct, total = neighbor_stats(result, shr.answer_key, shr.rows,
                                    shr.cols)
    acc = correct / total
    lat = [c.latency_ms for c in result.calls]
    tokens = sum(c.input_tokens for c in result.calls)
    print(f"Neighbor accuracy: {acc:.1%}  ({correct}/{total} neighbor "
          f"picks correct, {args.rows * args.cols} pieces)")
    print(f"Average latency:   {sum(lat) / len(lat):.0f} ms "
          f"over {len(lat)} calls")
    print(f"Total wall time:   {wall:.1f} s, {tokens} input tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
