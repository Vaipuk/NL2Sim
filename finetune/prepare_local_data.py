"""
finetune/prepare_local_data.py
-------------------------------
Writes train/val JSONL for local fine-tuning, keeping the evaluation test set
out of training.

The dataset ships three overlapping train/val splits. Their train portions
cover 875 of the 1000 samples; the other 125 appear in no train split, and
run_eval.py --dataset draws its cases from exactly those. So:

    875 samples -> train (90%) + val (10%)
    125 samples -> never touched, reserved for evaluation

Records are already in the {"messages": [user, assistant]} form that TRL reads
as a conversational dataset, so they are copied through unchanged.

    python finetune/prepare_local_data.py
    python finetune/prepare_local_data.py --out-dir outputs/ft_data --val-frac 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "outputs" / "data_gen" / "dataset"


def _sid(user_msg: str) -> str:
    return hashlib.md5(user_msg.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train/val JSONL that excludes the eval test set.")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "ft_data"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trained: set[str] = set()
    for f in sorted(DATASET_DIR.glob("*train*.jsonl")):
        trained |= {_sid(json.loads(l)["messages"][0]["content"]) for l in f.open(encoding="utf-8")}
    if not trained:
        raise SystemExit(f"no *train*.jsonl found in {DATASET_DIR}")

    usable, held_out = [], []
    for line in (DATASET_DIR / "dataset.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        (usable if _sid(rec["messages"][0]["content"]) in trained else held_out).append(rec)

    random.Random(args.seed).shuffle(usable)
    n_val = max(1, int(len(usable) * args.val_frac))
    val, train = usable[:n_val], usable[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test_heldout", held_out)):
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:13s} {len(rows):4d}  -> {p}")

    # The test set must share nothing with training, or the comparison is void.
    tr_ids = {_sid(r["messages"][0]["content"]) for r in train + val}
    te_ids = {_sid(r["messages"][0]["content"]) for r in held_out}
    assert not (tr_ids & te_ids), "train/val overlaps the held-out test set"
    print(f"\n  no overlap between train+val ({len(tr_ids)}) and test ({len(te_ids)})")


if __name__ == "__main__":
    main()
