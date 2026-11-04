"""
scripts/run_eval.py
--------------------
Runs every test description through the pipeline and prints one results table.

For each description it generates a config, counts blocking verification
issues (free, no API call), and computes the reliability score (one extra
API call for the NL reconstruction).

    python run_eval.py                          # all scripts/test_description*.txt
    python run_eval.py --no-score               # generate + verify only, no 2nd call
    python run_eval.py --rescore                # reuse saved configs, scoring only
    python run_eval.py --provider openai        # override $LLM_PROVIDER for this run
    python run_eval.py -d test_description.txt test_description3.txt

Results land in <outdir>/results.json alongside each generated config, so two
runs with different providers can be compared afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_client import resolve_provider, active_model          # noqa: E402
from nl_to_json import generate_json                            # noqa: E402
from repair_orchestrator import run_all_verifications           # noqa: E402


def blocking_count(cfg: dict) -> tuple[int, int]:
    """(blocking, warnings) from all three verification layers."""
    issues = run_all_verifications(cfg)
    blocking = sum(1 for i in issues
                   if getattr(i.severity, "name", str(i.severity)) == "BLOCKING")
    return blocking, len(issues) - blocking


def run_one(desc_path: Path, outdir: Path, provider: str | None,
            use_context: bool, do_score: bool, rescore: bool) -> dict:
    row: dict = {"description": desc_path.name}
    cfg_path = outdir / f"{desc_path.stem}.json"
    text = desc_path.read_text(encoding="utf-8").strip()

    # ---- generate (or reuse) ----
    t0 = time.time()
    if rescore:
        if not cfg_path.exists():
            row["error"] = f"no saved config at {cfg_path.name}"
            return row
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        row["gen_s"] = None
    else:
        cfg = generate_json(text, use_context=use_context, provider=provider)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        row["gen_s"] = round(time.time() - t0, 1)

    # ---- verify (free) ----
    row["blocking"], row["warnings"] = blocking_count(cfg)

    # ---- reliability score (one more API call) ----
    if do_score:
        from score_reliability import compute_reliability_score      # heavy import
        t1 = time.time()
        res = compute_reliability_score(text, cfg, provider=provider)
        row["score_s"] = round(time.time() - t1, 1)
        for k in ("bleu_1", "bleu_4", "numeric_f1", "numeric_precision",
                  "numeric_recall", "reliability_score"):
            row[k] = res[k]
        row["diagnosis"] = res["diagnosis"]
        (outdir / f"{desc_path.stem}_score.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
    return row


HEADERS = [("description", 28, "s"), ("blocking", 9, "d"), ("bleu_4", 8, ".4f"),
           ("numeric_f1", 11, ".4f"), ("numeric_precision", 10, ".4f"),
           ("numeric_recall", 8, ".4f"), ("reliability_score", 12, ".4f"),
           ("gen_s", 7, ".1f"), ("score_s", 8, ".1f")]
LABELS = {"description": "description", "blocking": "blocking", "bleu_4": "BLEU-4",
          "numeric_f1": "Num F1", "numeric_precision": "Num P",
          "numeric_recall": "Num R", "reliability_score": "reliability",
          "gen_s": "gen s", "score_s": "score s"}


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 112)
    print("  " + "".join(f"{LABELS[k]:<{w}}" for k, w, _ in HEADERS))
    print("  " + "-" * 110)
    for r in rows:
        if "error" in r:
            print(f"  {r['description']:<28}FAILED: {r['error']}")
            continue
        cells = []
        for k, w, fmt in HEADERS:
            v = r.get(k)
            text = "-" if v is None else (v if isinstance(v, str) else format(v, fmt))
            cells.append(f"{text:<{w}}")
        print("  " + "".join(cells))
    scored = [r for r in rows if "reliability_score" in r]
    if scored:
        print("  " + "-" * 110)
        n = len(scored)
        means = {k: sum(r[k] for r in scored) / n
                 for k in ("bleu_4", "numeric_f1", "numeric_precision",
                           "numeric_recall", "reliability_score")}
        blk = sum(r["blocking"] for r in rows if "blocking" in r) / max(
            1, len([r for r in rows if "blocking" in r]))
        print(f"  {'MEAN of ' + str(n):<28}{blk:<9.1f}{means['bleu_4']:<8.4f}"
              f"{means['numeric_f1']:<11.4f}{means['numeric_precision']:<10.4f}"
              f"{means['numeric_recall']:<8.4f}{means['reliability_score']:<12.4f}")
    print("=" * 112)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run every test description and tabulate the results.")
    ap.add_argument("-d", "--descriptions", nargs="+",
                    help="description files (default: scripts/test_description*.txt)")
    ap.add_argument("--outdir", default=None,
                    help="where configs and results land (default: outputs/eval_<provider>)")
    ap.add_argument("--provider", choices=("openai", "azure", "local"),
                    help="override $LLM_PROVIDER for this run")
    ap.add_argument("--no-context", action="store_true",
                    help="omit the system instructions (fewer tokens)")
    ap.add_argument("--no-score", action="store_true",
                    help="skip the reliability score, so one API call per description")
    ap.add_argument("--rescore", action="store_true",
                    help="reuse configs already in outdir; score only")
    args = ap.parse_args()

    provider = args.provider or resolve_provider()
    descs = ([Path(d) for d in args.descriptions] if args.descriptions
             else sorted(HERE.glob("test_description*.txt")))
    descs = [d if d.exists() else HERE / d.name for d in descs]
    missing = [d for d in descs if not d.exists()]
    if missing:
        sys.exit(f"not found: {', '.join(str(m) for m in missing)}")

    outdir = Path(args.outdir) if args.outdir else (
        HERE.parent / "outputs" / f"eval_{provider}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  provider  : {active_model(provider)}")
    print(f"  context   : {'off' if args.no_context else 'on'}")
    print(f"  scoring   : {'off' if args.no_score else 'on'}")
    print(f"  cases     : {len(descs)}")
    print(f"  outdir    : {outdir}")

    rows: list[dict] = []
    for i, d in enumerate(descs, 1):
        print(f"\n  [{i}/{len(descs)}] {d.name} ...", flush=True)
        try:
            row = run_one(d, outdir, args.provider, not args.no_context,
                          not args.no_score, args.rescore)
        except Exception as e:                      # one bad case must not kill the run
            row = {"description": d.name, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc(limit=2)
        rows.append(row)
        if "error" in row:
            print(f"        FAILED: {row['error']}")
        else:
            bits = [f"blocking={row['blocking']}"]
            if "reliability_score" in row:
                bits.append(f"reliability={row['reliability_score']:.2%}")
                bits.append(f"numR={row['numeric_recall']:.2f}")
            print("        " + "  ".join(bits))

    print_table(rows)
    summary = {"provider": provider, "model": active_model(provider),
               "context": not args.no_context, "results": rows}
    (outdir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  saved -> {outdir / 'results.json'}\n")


if __name__ == "__main__":
    main()
