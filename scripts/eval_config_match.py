"""
scripts/eval_config_match.py
-----------------------------
Scores a generated config against a benchmark config by key-value matching.

This is the comparison the papers describe being done by hand: "how closely
the output JSON matches the benchmark JSON". It is also WSC 2026's Algorithm 3
(KeyValueScore), which had no implementation.

Both configs are canonicalized first, so entity order never matters -- lists of
named entities are sorted by name, exactly as the repair loop does before
saving. Each config then flattens to a set of (path, value) pairs:

    supplier[Process Go].supplier_lead_time.parameters.a = 7

Note the path keys entities by NAME, not index, so inserting an extra supplier
shifts nothing. An entity the benchmark has and the output does not simply
contributes misses.

Four scores, reported separately because they fail differently:

    structure  did it produce the right fields at all (paths only)
    categorical did it get the string/enum values right
    numeric    did it get the numbers right (exact, or within --tol)
    overall    all leaf values together

Usage:
    python eval_config_match.py generated.json benchmark.json
    python eval_config_match.py generated.json benchmark.json --tol 0.05
    python eval_config_match.py gen_dir/ bench_dir/ --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_orchestrator import canonicalize_config, deep_sort  # noqa: E402

# Sections whose entries are identified by a "name" field. Paths through these
# key on the name so entity order and count never shift the comparison.
NAMED_SECTIONS = ("raw_materials", "intermediate_materials", "products",
                  "inventory", "supplier", "resource", "facility", "customer")

MISSING = "missing"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten(config: dict) -> dict:
    """Flatten a config to {path: leaf_value}. Named entities key by name."""
    out: dict = {}

    def walk(node, path: str):
        if isinstance(node, dict):
            for k, v in sorted(node.items()):
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            section = path.split(".")[0].split("[")[0]
            for idx, item in enumerate(node):
                if section in NAMED_SECTIONS and isinstance(item, dict) and isinstance(item.get("name"), str):
                    key = f"{path}[{item['name']}]"
                else:
                    key = f"{path}[{idx}]"
                walk(item, key)
        else:
            out[path] = node

    walk(deep_sort(canonicalize_config(config)), "")
    return out


def _values_match(a, b, tol: float) -> bool:
    if _is_num(a) and _is_num(b):
        if tol <= 0:
            return float(a) == float(b)
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1.0)
    return a == b


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score(generated: dict, benchmark: dict, tol: float = 0.0) -> dict:
    """Compare two configs. Returns per-category precision/recall/F1 plus a diff."""
    gen, ben = flatten(generated), flatten(benchmark)
    gpaths, bpaths = set(gen), set(ben)

    structure = _prf(len(gpaths & bpaths), len(gpaths - bpaths), len(bpaths - gpaths))

    buckets = {"categorical": [0, 0, 0], "numeric": [0, 0, 0], "overall": [0, 0, 0]}
    mismatches: list[dict] = []
    unspecified = invented = 0

    for path in bpaths:
        bv = ben[path]

        # A benchmark "missing" means the description never stated the field --
        # there is no ground truth to match, so scoring it measures nothing.
        # Over half of every benchmark config is this sentinel, mostly unused
        # distribution parameters (c/d/e on a two-parameter uniform), and
        # including them pinned categorical F1 near 0.5 regardless of output.
        # Counted instead: how many the model filled in anyway, which is the
        # hallucination signal and belongs on its own, not inside an F1.
        if bv == MISSING:
            unspecified += 1
            gv = gen.get(path)
            if gv is not None and gv != MISSING and gv != 0:
                invented += 1
                mismatches.append({"path": path, "expected": MISSING,
                                   "got": gv, "why": "invented"})
            continue

        kind = "numeric" if _is_num(bv) else "categorical"
        if path not in gen:
            buckets[kind][2] += 1          # fn: benchmark had it, output did not
            buckets["overall"][2] += 1
            mismatches.append({"path": path, "expected": bv, "got": None, "why": "absent"})
        elif _values_match(gen[path], bv, tol):
            buckets[kind][0] += 1
            buckets["overall"][0] += 1
        else:
            buckets[kind][1] += 1          # fp: present but wrong
            buckets["overall"][1] += 1
            why = "placeholder" if gen[path] == MISSING else "wrong value"
            mismatches.append({"path": path, "expected": bv, "got": gen[path], "why": why})

    for path in gpaths - bpaths:           # produced a field the benchmark lacks
        gv = gen[path]
        if gv == MISSING or gv == 0:       # schema padding, not a real claim
            continue
        kind = "numeric" if _is_num(gv) else "categorical"
        buckets[kind][1] += 1
        buckets["overall"][1] += 1
        mismatches.append({"path": path, "expected": None, "got": gv, "why": "extra"})

    return {
        "structure": structure,
        "categorical": _prf(*buckets["categorical"]),
        "numeric": _prf(*buckets["numeric"]),
        "overall": _prf(*buckets["overall"]),
        "benchmark_fields": len(bpaths),
        "generated_fields": len(gpaths),
        "scored_fields": len(bpaths) - unspecified,
        "unspecified": unspecified,
        "invented": invented,
        "mismatches": sorted(mismatches, key=lambda m: m["path"]),
    }


def print_report(result: dict, label: str = "") -> None:
    print("\n" + "=" * 62)
    print(f"CONFIG MATCH{(' - ' + label) if label else ''}")
    print("=" * 62)
    print(f"  benchmark fields : {result['benchmark_fields']}")
    print(f"  generated fields : {result['generated_fields']}\n")
    print(f"  {'':12s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
    for k in ("structure", "categorical", "numeric", "overall"):
        s = result[k]
        print(f"  {k:12s} {s['precision']:10.4f} {s['recall']:8.4f} {s['f1']:8.4f}")
    bad = result["mismatches"]
    if bad:
        print(f"\n  {len(bad)} mismatch(es):")
        for m in bad[:25]:
            print(f"    {m['why']:12s} {m['path']}")
            print(f"    {'':12s}   expected {m['expected']!r}, got {m['got']!r}")
        if len(bad) > 25:
            print(f"    ... and {len(bad) - 25} more")
    print("=" * 62)


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a generated config against a benchmark config.")
    ap.add_argument("generated", help="generated config .json, or a directory of them")
    ap.add_argument("benchmark", help="benchmark config .json, or a directory with matching filenames")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="relative tolerance for numbers (default 0 = exact)")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = ap.parse_args()

    gen_p, ben_p = Path(args.generated), Path(args.benchmark)
    results: dict = {}

    if gen_p.is_dir():
        for g in sorted(gen_p.glob("*.json")):
            b = ben_p / g.name
            if not b.exists():
                print(f"  [skip] no benchmark for {g.name}")
                continue
            results[g.name] = score(_load(g), _load(b), args.tol)
            print_report(results[g.name], g.name)
        if results:
            mean = sum(r["overall"]["f1"] for r in results.values()) / len(results)
            print(f"\n  mean overall F1 over {len(results)} case(s): {mean:.4f}")
    else:
        results[gen_p.name] = score(_load(gen_p), _load(ben_p), args.tol)
        print_report(results[gen_p.name], gen_p.name)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  saved -> {args.json}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Self-check: no files needed.
        base = {
            "config_info": [{"name": "x", "version": "1"}],
            "raw_materials": [{"name": "chip"}],
            "products": [{"name": "CPU", "bom": {"chip": 1}}],
            "supplier": [{"name": "Acme", "supplier_cost": 200,
                          "supplier_lead_time": {"distribution": "constant",
                                                 "parameters": {"a": 7, "b": 0}}}],
            "customer": [{"name": "Ross", "shortage_policy": "salelost"}],
        }
        import copy

        same = score(base, copy.deepcopy(base))
        assert same["overall"]["f1"] == 1.0, same["overall"]
        assert not same["mismatches"]

        # entity order must not matter
        reordered = copy.deepcopy(base)
        reordered["supplier"].insert(0, {"name": "Zeta", "supplier_cost": 5})
        r = score(reordered, base)
        assert all(m["why"] == "extra" for m in r["mismatches"]), r["mismatches"]
        assert r["overall"]["recall"] == 1.0, "benchmark fields all still found"

        # a wrong number is numeric, a wrong enum is categorical
        wrong = copy.deepcopy(base)
        wrong["supplier"][0]["supplier_cost"] = 250
        wrong["customer"][0]["shortage_policy"] = "Sale_lost"
        r = score(wrong, base)
        assert r["numeric"]["fp"] == 1 and r["categorical"]["fp"] == 1, r
        assert r["overall"]["f1"] < 1.0

        # tolerance accepts a near miss
        near = copy.deepcopy(base)
        near["supplier"][0]["supplier_cost"] = 204
        assert score(near, base)["numeric"]["fp"] == 1
        assert score(near, base, tol=0.05)["numeric"]["fp"] == 0

        # "missing" is reported as a placeholder, not a wrong value
        miss = copy.deepcopy(base)
        miss["supplier"][0]["supplier_cost"] = "missing"
        assert [m["why"] for m in score(miss, base)["mismatches"]] == ["placeholder"]

        # A benchmark "missing" is not scored: over half of every real benchmark
        # is that sentinel (unused distribution slots), and scoring it pinned
        # categorical F1 near 0.5 whatever the model produced.
        pad = copy.deepcopy(base)
        pad["supplier"][0]["supplier_lead_time"]["parameters"].update(c="missing", d="missing")
        gen = copy.deepcopy(pad)
        gen["supplier"][0]["supplier_lead_time"]["parameters"].update(c=0, d=0)
        r = score(gen, pad)
        assert r["overall"]["f1"] == 1.0, r["overall"]
        assert r["unspecified"] == 2 and r["invented"] == 0, r

        # but filling an unspecified field with a real value is flagged
        gen2 = copy.deepcopy(pad)
        gen2["supplier"][0]["supplier_lead_time"]["parameters"]["c"] = 99
        r2 = score(gen2, pad)
        assert r2["invented"] == 1, r2
        assert [m["why"] for m in r2["mismatches"]] == ["invented"], r2["mismatches"]

        print("eval_config_match self-check OK (8 cases)")
    else:
        main()
