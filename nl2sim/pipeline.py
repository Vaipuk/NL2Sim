"""
nl2sim/pipeline.py
-------------------
NL2Sim pipeline library — importable by run_pipeline.py or any external code.

Usage (imported):
    from nl2sim.pipeline import Pipeline
    results = Pipeline(description).run()
    results = Pipeline(description).run_with_graph(image_path)
"""

from __future__ import annotations

import json
import sys
import os
import platform
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# ── ensure scripts/ is on path ─────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── ensure scripts/graph_output is on path ──────────────────
GRAPH_OUTPUT_DIR = SCRIPTS_DIR / "graph_output"
if str(GRAPH_OUTPUT_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_OUTPUT_DIR))

# ── ensure scripts/graph_input is on path ───────────────────
GRAPH_INPUT_DIR = SCRIPTS_DIR / "graph_input"
if str(GRAPH_INPUT_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_INPUT_DIR))

from nl_to_json        import generate_json
from repair_orchestrator      import run_repair_loop, canonicalize_config, deep_sort
from score_reliability import compute_reliability_score, print_score_report
from simulate          import run_simulation
from nl_to_whatif      import generate_whatif
from what_if_engine    import apply_what_if_config, WhatIfError
from nl2sim_graph_output import (
    build_graph, assign_layout_positions,
    separate_parallel_edges, render_html,
)
from image_to_json import extract_topology
from graph_only_to_json import (
    extract_full_config, collect_refinement_answer,
    interpret_refinement_answer, apply_updates,
)


def _open_in_browser(path: Path) -> None:
    """
    Opens a local HTML file in the default browser. Uses the platform's
    native opener (more reliable than webbrowser.open, which can return
    True on macOS without actually launching anything from some shell
    contexts) with a webbrowser fallback for other platforms.
    """
    resolved = str(path.resolve())
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", resolved], check=True)
        elif system == "Windows":
            os.startfile(resolved)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", resolved], check=True)
        return
    except Exception:
        pass

    if not webbrowser.open(f"file://{resolved}"):
        print("  (Could not auto-open browser — open the file manually.)")


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """
    Full NL2Sim pipeline:
      1. Generate JSON from natural language description
         (or: reconcile a graph sketch + description via merge.py)
      2. Validate and repair JSON
      3. Render scenario graph and confirm with user
      4. Compute reliability score
      5. Run simulation
      6. Optional: what-if analysis
    """

    def __init__(
        self,
        description:  str,
        output_dir:   Optional[Path] = None,
        use_context:  bool = True,
        simulate:     bool = True,
        use_azure:    bool = False,
    ):
        self.description = description
        self.use_context = use_context
        self.simulate    = simulate
        # use_azure=True forces Azure; otherwise $LLM_PROVIDER picks the
        # backend (openai | azure | local), defaulting to openai.
        self.use_azure   = use_azure

        # ── output directory ───────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base      = output_dir or (
            Path(__file__).resolve().parents[1] / "outputs"
        )
        self.output_dir = Path(base) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── state ──────────────────────────────────────────
        self.config:             Optional[Dict[str, Any]] = None
        self.reliability_result: Optional[Dict[str, Any]] = None
        self.simulation_result:  Optional[Dict[str, Any]] = None
        self.whatif_results:     list = []
        self.whatif_count:       int  = 0
        # True once a graph/sketch was part of this run's source (either
        # graph-only or graph+text) -- set by generate_from_graph() /
        # generate_from_graph_only(). Used to skip the reliability score
        # step, since it measures fidelity against a NL description and
        # isn't a meaningful signal when the source wasn't purely text.
        self.used_graph:         bool = False

    # ── Step 1 — Generate ──────────────────────────────────

    def generate(self) -> Dict[str, Any]:
        print("\nStep 1/5 — Generating JSON from description...")
        cfg = generate_json(
            self.description,
            use_context=self.use_context,
            use_azure=self.use_azure,
        )
        self._save("config_raw.json", cfg)
        print(f"  ✓ JSON generated → {self.output_dir / 'config_raw.json'}")
        return cfg

    # ── Step 1 (graph variant) — Generate from graph + text ────

    def generate_from_graph(self, image_path: str) -> Dict[str, Any]:
        """
        Graph + Text entry: extracts topology from a sketch/graph image,
        generates the NL-derived config from self.description, then
        reconciles the two via scripts/graph_input/merge.py (node
        matching, facility typing, edge resolution, material detection).

        Returns the reconciled config, ready for validate() onward. Used
        by run_with_graph() in place of generate() when a graph image is
        supplied alongside the description.
        """
        self.used_graph = True
        print("\nStep 1/5 — Extracting topology from sketch...")
        topology = extract_topology(image_path)
        self._save("graph_topology.json", topology)
        graph_path = self.output_dir / "graph_topology.json"
        print(f"  ✓ Topology extracted → {graph_path}")

        print("\nStep 1/5 — Generating NL-derived config from description...")
        nl_cfg = generate_json(
            self.description,
            use_context=self.use_context,
            use_azure=self.use_azure,
        )
        self._save("nl_config.json", nl_cfg)
        nl_path = self.output_dir / "nl_config.json"
        print(f"  ✓ NL config generated → {nl_path}")

        print("\nStep 1/5 — Reconciling graph + NL sources (merge.py)...")
        merge_out = self.output_dir / "nl_fixed.json"
        merge_script = GRAPH_INPUT_DIR / "merge.py"
        result = subprocess.run(
            [sys.executable, str(merge_script),
             "--graph", str(graph_path),
             "--nl", str(nl_path),
             "--out", str(merge_out)],
        )
        if result.returncode != 0:
            raise SystemExit("merge.py did not complete successfully — aborting.")

        with open(merge_out, "r", encoding="utf-8") as f:
            reconciled_cfg = json.load(f)

        print(f"  ✓ Reconciled config → {merge_out}")
        return reconciled_cfg

    # ── Step 1 (graph-only variant) — Generate from graph alone ────

    def generate_from_graph_only(self, image_path: str, max_rounds: int = 4) -> Dict[str, Any]:
        """
        Graph-only entry: no description exists to reconcile against, so
        this extracts a full NL2Sim config directly from the sketch/graph
        image via graph_only_to_json.py's schema-aware prompt, then runs
        its single-source clarification loop (question -> answer -> fill)
        for anything the model wasn't confident enough to extract outright.

        Returns the resulting config, ready for validate() onward. Used
        by run_with_graph_only() in place of generate().
        """
        self.used_graph = True
        print("\nStep 1/5 — Extracting config from sketch...")
        result = extract_full_config(image_path)
        config = result.get("config", {})
        questions = result.get("clarification_questions", [])
        self._save("graph_only_config_initial.json", config)
        print(f"  ✓ Initial extraction complete → {self.output_dir / 'graph_only_config_initial.json'}")

        for round_num in range(1, max_rounds + 1):
            if not questions:
                break
            print(f"\n--- Clarification round {round_num} ---")
            answer = collect_refinement_answer(questions)
            if not answer:
                print("  No answer given — leaving remaining questions unresolved.")
                break

            result = interpret_refinement_answer(config, questions, answer)
            updates = result.get("updates", [])
            still_unresolved_ids = set(result.get("still_unresolved_ids", []))

            log = apply_updates(config, updates)
            print("\n  Applied:")
            for line in log:
                print(f"    {line}")

            questions = [q for q in questions if q["id"] in still_unresolved_ids]
        else:
            if questions:
                print(f"\n  Reached max rounds ({max_rounds}) with {len(questions)} question(s) still unresolved.")

        if questions:
            print(f"\n  {len(questions)} field(s) remain 'missing' with unresolved questions:")
            for q in questions:
                print(f'    [{q["id"]}] {q["question"]}')
        else:
            print("\n  All clarification questions resolved.")

        self._save("graph_only_config_final.json", config)
        print(f"  ✓ Final graph-only config → {self.output_dir / 'graph_only_config_final.json'}")
        return config

    # ── Step 2 — Validate + Repair ─────────────────────────

    def validate(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        print("\nStep 2/5 — Validating and repairing JSON...")

        repaired, remaining = run_repair_loop(cfg, max_iterations=60, verbose=True, description=self.description)
        if remaining:
            print(f"\n  {len(remaining)} issue(s) could not be fully resolved:")
            for issue in remaining:
                print("   ", issue)

        repaired = deep_sort(canonicalize_config(repaired))
        self._save("config.json", repaired)
        print(f"  ✓ Config validated → {self.output_dir / 'config.json'}")
        return repaired

    # ── Step 3 — Graph Review ──────────────────────────────

    def review_graph(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Renders the validated config as an interactive graph and lets the
        user confirm before scoring, or hand-edit the config and
        re-validate. Runs after verification/repair (Step 2), before
        reliability scoring (Step 4).
        """
        print("\nStep 3/5 — Rendering scenario graph for review...")
        graph_path = self.output_dir / "scenario_graph.html"

        def _render(c: Dict[str, Any]) -> None:
            G = build_graph(c)
            assign_layout_positions(G)
            separate_parallel_edges(G)
            render_html(G, str(graph_path))
            print(f"  ✓ Graph saved → {graph_path}")
            _open_in_browser(graph_path)

        _render(cfg)

        while True:
            print("\n" + "─" * 60)
            print("Review the graph, then choose how to proceed:")
            print("  1) Proceed to scoring")
            print("  2) Retry (enter new description or select a .txt file)")
            print("  3) Abort")
            print("─" * 60)
            choice = input("Select option: ").strip()

            if choice == "1":
                return cfg

            elif choice == "2":
                print("\n" + "─" * 60)
                print("How would you like to provide the new description?")
                print("  1) Type it now")
                print("  2) Select a .txt file")
                print("─" * 60)
                sub_choice = input("Select option: ").strip()

                if sub_choice == "1":
                    print("\nEnter new supply chain description.")
                    print("Press Enter on an empty line when done.\n")
                    lines = []
                    while True:
                        line = input()
                        if line == "":
                            break
                        lines.append(line)
                    new_description = "\n".join(lines).strip()

                elif sub_choice == "2":
                    file_path = input("\n  Path to .txt file: ").strip()
                    try:
                        new_description = Path(file_path).read_text(encoding="utf-8").strip()
                    except OSError as e:
                        print(f"  ✗ Could not read file: {e}")
                        continue

                else:
                    print("  Invalid selection. Please enter 1 or 2.")
                    continue

                if not new_description:
                    print("  No description provided — keeping original.")
                    continue

                self.description = new_description
                print("\n  Regenerating from description...")
                cfg = self.generate()
                cfg = self.validate(cfg)
                self.config = cfg
                print("\nStep 3/5 — Re-rendering scenario graph for review...")
                _render(cfg)

            elif choice == "3":
                raise SystemExit("Aborted by user at graph review.")

            else:
                print("Invalid selection. Please enter 1, 2 or 3.")

    # ── Step 4 — Reliability Score ─────────────────────────

    def maybe_score(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decides whether to run the reliability score at all, and how.
        The score measures how closely the generated config matches a
        natural-language description -- that comparison isn't meaningful
        when a graph/sketch was part of the source (self.used_graph),
        since there may be no description, or the description alone
        never fully specified the topology to begin with. In that case,
        scoring is skipped automatically with an explanatory note.

        For a text-only run, scoring is still meaningful, but not
        everyone wants to pay for it every time -- offer the choice
        rather than forcing it.
        """
        if self.used_graph:
            print("\n" + "─" * 60)
            print("Skipping reliability score: it measures how closely the config")
            print("matches a natural-language description, which isn't a meaningful")
            print("signal when a graph/sketch was part of the source. Reliability")
            print("scoring is only accurate when natural-language instructions are")
            print("provided as the source.")
            print("─" * 60)
            return {}

        print("\n" + "─" * 60)
        print("Reliability score compares the generated config against your")
        print("description -- it takes a bit of extra time and API cost.")
        print("  1) Compute reliability score")
        print("  2) Skip scoring and proceed to simulation")
        print("─" * 60)
        while True:
            choice = input("Select option: ").strip()
            if choice == "1":
                return self.score(cfg)
            elif choice == "2":
                print("  Skipping reliability score.")
                return {}
            else:
                print("Invalid selection. Please enter 1 or 2.")

    def score(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        print("\nStep 4/5 — Computing reliability score...")
        try:
            result = compute_reliability_score(
                self.description,
                cfg,
                use_azure=self.use_azure,
            )
            print_score_report(result)
            self._save("reliability_score.json", result)
            return result
        except Exception as e:
            print(f"\n  ✗ Reliability score failed: {e}")
            print("\n  How would you like to proceed?")
            print("  1) Skip score and continue")
            print("  2) Abort")
            while True:
                choice = input("  Select option #: ").strip()
                if choice == "1":
                    print("  Skipping reliability score.")
                    return {}
                elif choice == "2":
                    raise SystemExit("Aborted by user.")
                else:
                    print("  Invalid selection. Please enter 1 or 2.")

    # ── Step 5 — Simulate ──────────────────────────────────

    def simulate_run(
        self,
        cfg: Dict[str, Any],
        output_name: str = "simulation_results.json",
    ) -> Dict[str, Any]:
        print(f"\nStep 5/5 — Running simulation...")
        out_path = str(self.output_dir / output_name)
        result   = run_simulation(cfg, output_path=out_path)
        print(f"  ✓ Results saved → {out_path}")
        return result

    # ── Step 6 — What-if ───────────────────────────────────

    def run_whatif(
        self,
        cfg:          Dict[str, Any],
        instruction:  str,
        use_examples: bool = True,
    ) -> Dict[str, Any]:
        self.whatif_count += 1
        n = self.whatif_count

        print(f"\n  Generating what-if JSON...")
        whatif_json = generate_whatif(
            instruction, cfg, use_examples=use_examples)
        self._save(f"whatif_{n}_changes.json", whatif_json)

        print(f"  Applying changes...")
        from copy import deepcopy
        modified = apply_what_if_config(deepcopy(cfg), whatif_json)

        # Bump the config version for this what-if variant -- base runs
        # are always "1.0" (auto-assigned, never asked); each what-if
        # increments by 0.1 (1.1, 1.2, ...) using whatif_count as the
        # counter, so the version always reflects how many what-if
        # iterations produced this particular config.
        if isinstance(modified.get("config_info"), list) and modified["config_info"]:
            modified["config_info"][0]["version"] = f"1.{n}"

        print(f"  Validating modified config...")
        repaired, remaining = run_repair_loop(modified, max_iterations=60, verbose=True, description=self.description)
        if remaining:
            print(f"\n  {len(remaining)} issue(s) could not be fully resolved:")
            for issue in remaining:
                print("   ", issue)
        repaired = deep_sort(canonicalize_config(repaired))
        self._save(f"whatif_{n}_config.json", repaired)
        print(f"  ✓ What-if config → {self.output_dir / f'whatif_{n}_config.json'}")

        print(f"  Running what-if simulation...")
        out_path = str(self.output_dir / f"whatif_{n}_results.json")
        result   = run_simulation(repaired, output_path=out_path)
        print(f"  ✓ What-if results → {out_path}")

        self.whatif_results.append({
            "instruction": instruction,
            "whatif_json": whatif_json,
            "result":      result,
        })

        return repaired, result

    # ── Full pipeline ──────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        self._print_header()

        # Step 1 — Generate
        cfg = self.generate()

        # Step 2 — Validate
        cfg = self.validate(cfg)
        self.config = cfg

        # Step 3 — Graph review
        cfg = self.review_graph(cfg)
        self.config = cfg

        # Step 4 — Score
        self.reliability_result = self.maybe_score(cfg)

        # ── Ask user to proceed or re-enter description ────
        while True:
            print("\n" + "─" * 60)
            print("How would you like to proceed?")
            print("  1) Proceed to simulation")
            print("  2) Re-enter description and regenerate")
            print("  3) Exit")
            print("─" * 60)
            choice = input("Select option: ").strip()

            if choice == "1":
                break

            elif choice == "2":
                print("\nEnter new supply chain description.")
                print("Press Enter on an empty line when done.\n")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                new_description = "\n".join(lines).strip()
                if not new_description:
                    print("No description entered — keeping original.")
                    break
                self.description = new_description
                print("  ✓ New description received. Regenerating...\n")

                # regenerate + revalidate + review + rescore
                cfg = self.generate()
                cfg = self.validate(cfg)
                self.config = cfg
                cfg = self.review_graph(cfg)
                self.config = cfg
                self.reliability_result = self.maybe_score(cfg)

            elif choice == "3":
                print("Exiting pipeline.")
                sys.exit(0)

            else:
                print("Invalid selection. Please enter 1, 2 or 3.")

        # Step 5 — Simulate
        if self.simulate:
            self.simulation_result = self.simulate_run(cfg)

        # Step 6 — What-if loop
        self._whatif_loop(cfg)

        # Save summary
        summary = self._build_summary()
        self._save("pipeline_summary.json", summary)

        self._print_footer()
        return summary

    # ── Validate + Simulate (skip LLM step) ───────────────

    def validate_and_simulate(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point when skipping JSON generation (--from-config flag, or
        a config already produced by generate_from_graph()).
        Starts from an existing config JSON.
        """
        self._print_header()

        # Step 2 — Validate
        cfg = self.validate(cfg)
        self.config = cfg

        # Step 3 — Graph review
        cfg = self.review_graph(cfg)
        self.config = cfg

        # Step 4 — Score
        self.reliability_result = self.maybe_score(cfg)

        # ── Ask user to proceed ────────────────────────────
        while True:
            print("\n" + "─" * 60)
            print("How would you like to proceed?")
            print("  1) Proceed to simulation")
            print("  2) Exit")
            print("─" * 60)
            choice = input("Select option: ").strip()

            if choice == "1":
                break
            elif choice == "2":
                print("Exiting pipeline.")
                sys.exit(0)
            else:
                print("Invalid selection. Please enter 1 or 2.")

        # Step 5 — Simulate
        if self.simulate:
            self.simulation_result = self.simulate_run(cfg)

        # Step 6 — What-if loop
        self._whatif_loop(cfg)

        # Save summary
        summary = self._build_summary()
        self._save("pipeline_summary.json", summary)

        self._print_footer()
        return summary

    # ── Graph + Text entry point ───────────────────────────

    def run_with_graph(self, image_path: str) -> Dict[str, Any]:
        """
        Full pipeline starting from a graph sketch/image PLUS the
        description already passed to the constructor. Reconciles the two
        via generate_from_graph() (topology extraction + NL generation +
        merge.py), then continues through validate_and_simulate() exactly
        as the --from-config path does.
        """
        cfg = self.generate_from_graph(image_path)
        return self.validate_and_simulate(cfg)

    def run_with_graph_only(self, image_path: str) -> Dict[str, Any]:
        """
        Full pipeline starting from a graph sketch/image alone, no
        description. Uses generate_from_graph_only() (schema-aware
        extraction + single-source clarification loop), then continues
        through validate_and_simulate() exactly as the other entry points
        do. Verification/repair runs unchanged for now -- whether the
        layers need reordering or a graph-only-specific pass is a
        follow-up decision, not handled here yet.
        """
        cfg = self.generate_from_graph_only(image_path)
        return self.validate_and_simulate(cfg)

    # ── What-if interactive loop ───────────────────────────

    def _whatif_loop(self, cfg: Dict[str, Any]):
        current_cfg = cfg
        while True:
            print("\n" + "=" * 60)
            answer = input(
                "Would you like to run a what-if analysis? (yes/no): "
            ).strip().lower()

            if answer not in {"yes", "y", "1"}:
                break

            # ── ask about examples ─────────────────────────
            print("\n" + "─" * 60)
            print("Few-shot examples improve accuracy but use more tokens.")
            print("Recommended: Yes for complex changes, No for simple ones.")
            print("─" * 60)
            ex = input(
                "Include examples? (yes/no) [default: yes]: "
            ).strip().lower()
            use_examples = ex not in {"no", "n"}

            # ── get instruction ────────────────────────────
            print("\n" + "=" * 60)
            print("NL2Sim What-If Instruction")
            print("=" * 60)
            print("Describe the change you want to make.")
            print("⚠️  Be specific — mention exact entity names.")
            print("Press Enter on an empty line when done.\n")

            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            instruction = "\n".join(lines).strip()
            if not instruction:
                print("No instruction entered — skipping.")
                continue

            # ── save instruction ───────────────────────────
            n         = self.whatif_count + 1
            instr_dir = self.output_dir / "whatif_instructions"
            instr_dir.mkdir(exist_ok=True)
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            (instr_dir / f"instruction_{n}_{ts}.txt").write_text(
                f"Instruction {n}:\n{instruction}\n",
                encoding="utf-8",
            )

            # ── run what-if ────────────────────────────────
            try:
                current_cfg, _ = self.run_whatif(
                    current_cfg, instruction,
                    use_examples=use_examples,
                )
            except WhatIfError as e:
                print(f"  ✗ What-if failed: {e}")

    # ── Helpers ────────────────────────────────────────────

    def _save(self, filename: str, data: dict):
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _build_summary(self) -> Dict[str, Any]:
        return {
            "description":        self.description[:500],
            "output_dir":         str(self.output_dir),
            "reliability_score":  self.reliability_result,
            "simulation_result":  self.simulation_result,
            "whatif_count":       self.whatif_count,
            "whatif_results": [
                {
                    "instruction": w["instruction"],
                    "whatif_json": w["whatif_json"],
                }
                for w in self.whatif_results
            ],
        }

    def _print_header(self):
        print("\n" + "=" * 60)
        print("NL2Sim Pipeline")
        print("=" * 60)

    def _print_footer(self):
        print("\n" + "=" * 60)
        print("Pipeline complete.")
        print(f"All outputs saved → {self.output_dir}")
        print("=" * 60)