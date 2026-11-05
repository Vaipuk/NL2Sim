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
from eval_config_match import score as match_score              # noqa: E402

DATASET_DIR = HERE.parent / "outputs" / "data_gen" / "dataset"


def _sample_id(user_msg: str) -> str:
    import hashlib
    return hashlib.md5(user_msg.encode()).hexdigest()


def split_user_message(u: str) -> str:
    """
    Pull the description out of a training sample's user message, which is
    "<instruction prefix>\\n\\n<description>\\n\\n<schema JSON>". Ten different
    prefixes are used across the dataset; all end at the first blank line, and
    the schema starts at the first brace.
    """
    head = u[:u.find("{")]
    parts = head.split("\n\n", 1)
    return (parts[1] if len(parts) > 1 else parts[0]).strip()


def held_out_cases(n: int, seed: int) -> list[tuple[str, str, dict]]:
    """
    (name, description, benchmark_config) for samples that appear in NO
    training split, so the comparison is fair to a model fine-tuned on any of
    them. Qwen has seen none of the dataset, but the fine-tuned GPT has.
    """
    import random

    trained: set[str] = set()
    for f in DATASET_DIR.glob("*train*.jsonl"):
        trained |= {_sample_id(json.loads(l)["messages"][0]["content"])
                    for l in f.open(encoding="utf-8")}

    pool = []
    for idx, line in enumerate(( DATASET_DIR / "dataset.jsonl").open(encoding="utf-8")):
        msgs = json.loads(line)["messages"]
        if _sample_id(msgs[0]["content"]) in trained:
            continue
        pool.append((f"sample_{idx:04d}", split_user_message(msgs[0]["content"]),
                     json.loads(msgs[-1]["content"])))
    if not pool:
        sys.exit("no held-out samples found -- is outputs/data_gen/dataset populated?")
    random.Random(seed).shuffle(pool)
    return pool[:n]


def verify_counts(cfg: dict) -> dict:
    """
    Verification issues, split by what they mean about the model.

    MISSING_REQUIRED_VALUE means the description never stated the field and the
    model correctly declined to invent one -- the "missing" sentinel doing its
    job. Counting it as a defect ranks a terse description as the worst output
    when it may be among the best, so it is reported separately from errors the
    model is actually responsible for.
    """
    issues = run_all_verifications(cfg)
    counts = {"blocking": 0, "warnings": 0, "incomplete": 0, "wrong": 0}
    by_type: dict = {}
    for i in issues:
        sev = getattr(i.severity, "name", str(i.severity))
        counts["blocking" if sev == "BLOCKING" else "warnings"] += 1
        name = getattr(i.defect_type, "name", str(i.defect_type))
        by_type[name] = by_type.get(name, 0) + 1
        counts["incomplete" if name == "MISSING_REQUIRED_VALUE" else "wrong"] += 1
    counts["by_defect_type"] = by_type
    return counts


def run_one(name: str, text: str, outdir: Path, provider: str | None,
            use_context: bool, do_score: bool, rescore: bool,
            benchmark: dict | None = None) -> dict:
    row: dict = {"description": name}
    cfg_path = outdir / f"{Path(name).stem}.json"

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
    row.update(verify_counts(cfg))

    # ---- key-value match against the benchmark, when we have one ----
    if benchmark is not None:
        m = match_score(cfg, benchmark)
        row["kv_overall"] = m["overall"]["f1"]
        row["kv_structure"] = m["structure"]["f1"]
        row["kv_numeric"] = m["numeric"]["f1"]
        row["kv_categorical"] = m["categorical"]["f1"]
        row["kv_mismatches"] = len(m["mismatches"])
        row["kv_invented"] = m["invented"]
        row["kv_scored_fields"] = m["scored_fields"]
        (outdir / f"{Path(name).stem}_match.json").write_text(
            json.dumps(m, indent=2), encoding="utf-8")

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
        (outdir / f"{Path(name).stem}_score.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
    return row


HEADERS = [("description", 24, "s"), ("kv_overall", 8, ".4f"), ("kv_numeric", 8, ".4f"),
           ("kv_structure", 10, ".4f"), ("kv_invented", 9, "d"), ("wrong", 7, "d"), ("incomplete", 12, "d"),
           ("blocking", 9, "d"), ("bleu_4", 8, ".4f"), ("numeric_f1", 8, ".4f"),
           ("numeric_precision", 8, ".4f"), ("numeric_recall", 8, ".4f"),
           ("reliability_score", 12, ".4f"), ("gen_s", 7, ".1f")]
LABELS = {"description": "case", "kv_overall": "KV all", "kv_numeric": "KV num",
          "kv_structure": "KV struct", "kv_categorical": "KV categ",
          "kv_invented": "invented",
          "wrong": "WRONG",
          "incomplete": "not stated", "blocking": "blocking", "bleu_4": "BLEU-4",
          "numeric_f1": "Num F1", "numeric_precision": "Num P",
          "numeric_recall": "Num R", "reliability_score": "reliability",
          "gen_s": "gen s", "score_s": "score s"}


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 112)
    print("  " + "".join(f"{LABELS[k]:<{w}}" for k, w, _ in HEADERS))
    print("  " + "-" * 110)
    for r in rows:
        if "error" in r:
            print(f"  {r['description']:<24}FAILED: {r['error']}")
            continue
        cells = []
        for k, w, fmt in HEADERS:
            v = r.get(k)
            text = "-" if v is None else (v if isinstance(v, str) else format(v, fmt))
            cells.append(f"{text:<{w}}")
        print("  " + "".join(cells))
    kv = [r for r in rows if "kv_overall" in r]
    if kv:
        print("  " + "-" * 110)
        for k in ("kv_overall", "kv_numeric", "kv_structure", "kv_categorical"):
            print(f"  mean {LABELS.get(k, k):<10s} {sum(r[k] for r in kv) / len(kv):.4f}"
                  f"   over {len(kv)} case(s)")

    scored = [r for r in rows if "reliability_score" in r]
    if scored:
        print("  " + "-" * 110)
        n = len(scored)
        means = {k: sum(r[k] for r in scored) / n
                 for k in ("bleu_4", "numeric_f1", "numeric_precision",
                           "numeric_recall", "reliability_score")}
        ver = [r for r in rows if "wrong" in r]
        mw = sum(r["wrong"] for r in ver) / max(1, len(ver))
        mi = sum(r["incomplete"] for r in ver) / max(1, len(ver))
        mb = sum(r["blocking"] for r in ver) / max(1, len(ver))
        print(f"  {'MEAN of ' + str(n):<24}{mw:<7.1f}{mi:<12.1f}{mb:<9.1f}"
              f"{means['bleu_4']:<8.4f}{means['numeric_f1']:<8.4f}"
              f"{means['numeric_precision']:<8.4f}{means['numeric_recall']:<8.4f}"
              f"{means['reliability_score']:<12.4f}")
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
    ap.add_argument("--dataset", type=int, metavar="N",
                    help="evaluate N held-out dataset samples against their benchmark "
                         "configs (key-value match) instead of the test descriptions")
    ap.add_argument("--seed", type=int, default=0,
                    help="sample selection seed for --dataset (default 0)")
    args = ap.parse_args()

    provider = args.provider or resolve_provider()
    if args.dataset:
        cases = [(n, t, b) for n, t, b in held_out_cases(args.dataset, args.seed)]
    else:
        descs = ([Path(d) for d in args.descriptions] if args.descriptions
                 else sorted(HERE.glob("test_description*.txt")))
        descs = [d if d.exists() else HERE / d.name for d in descs]
        missing = [d for d in descs if not d.exists()]
        if missing:
            sys.exit(f"not found: {', '.join(str(m) for m in missing)}")
        cases = [(d.name, d.read_text(encoding="utf-8").strip(), None) for d in descs]

    tag = f"dataset{args.dataset}" if args.dataset else "cases"
    outdir = Path(args.outdir) if args.outdir else (
        HERE.parent / "outputs" / f"eval_{provider}_{tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  provider  : {active_model(provider)}")
    print(f"  context   : {'off' if args.no_context else 'on'}")
    print(f"  scoring   : {'off' if args.no_score else 'on'}")
    print(f"  cases     : {len(cases)}" + ("  (held-out dataset samples, benchmark available)" if args.dataset else ""))
    print(f"  outdir    : {outdir}")

    rows: list[dict] = []
    for i, (name, text, bench) in enumerate(cases, 1):
        print(f"\n  [{i}/{len(cases)}] {name} ...", flush=True)
        try:
            row = run_one(name, text, outdir, args.provider, not args.no_context,
                          not args.no_score, args.rescore, benchmark=bench)
        except Exception as e:                      # one bad case must not kill the run
            row = {"description": name, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc(limit=2)
        rows.append(row)
        if "error" in row:
            print(f"        FAILED: {row['error']}")
        else:
            bits = [f"wrong={row['wrong']}", f"not-stated={row['incomplete']}"]
            if "kv_overall" in row:
                bits.insert(0, f"kv={row['kv_overall']:.4f}")
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
