import json
import os
import re
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .expectations import ExpectationMixin  # added

# Rich console
console = Console(highlight=False)

# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[str]:
    """
    Try to extract JSON from common patterns:
    - fenced code blocks ```json { ... } ```
    - bare JSON somewhere in the text (best-effort)
    """
    if not isinstance(text, str):
        return None

    # Prefer fenced json code blocks first
    m = _JSON_BLOCK.search(text)
    if m:
        return m.group(1).strip()

    # Fallback: best-effort brute search for a top-level JSON object/array
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    first = min(starts) if starts else -1

    last_curly = text.rfind("}")
    last_brack = text.rfind("]")
    last = max(last_curly, last_brack)

    if first != -1 and last > first:
        candidate = text[first : last + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Suite defaults (env)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_RUNS = int(os.getenv("LLM_RUNS_PER_TEST", "1"))
DEFAULT_ACC_THRESHOLD = os.getenv("LLM_ACCURACY_THRESHOLD")
DEFAULT_ACC_THRESHOLD = float(DEFAULT_ACC_THRESHOLD) if DEFAULT_ACC_THRESHOLD else None

# Raw dump controls
RAW_DUMP_MODE = os.getenv("LLM_RAW_DUMP", "never").strip().lower()  # "never" | "fail" | "always"
RAW_DUMP_DIR = os.getenv("LLM_RAW_DUMP_DIR", "").strip()  # "" -> print to console

# Global results registry for optional suite summary
RESULTS: List[Dict[str, Any]] = []


# ──────────────────────────────────────────────────────────────────────────────
# Pretty output helpers
# ──────────────────────────────────────────────────────────────────────────────
def _truncate(s: str, n: int = 500) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _accuracy_bar(acc: float, width: int = 24) -> Text:
    filled = int(round(acc * width))
    empty = max(0, width - filled)
    bar = Text("█" * filled, style="green") + Text("░" * empty, style="grey39")
    pct = Text(f"  {acc*100:.1f}%", style="bold")
    return bar + pct


def _result_badge(ok: bool) -> Text:
    return Text(" PASS ", style="bold white on green") if ok else Text(" FAIL ", style="bold white on red")


def _threshold_badge(thr: Optional[float], acc: float) -> Text:
    if thr is None:
        return Text("")
    color = "green" if acc >= thr else "red"
    return Text(f"  (threshold {thr:.2f})", style=f"bold {color}")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario
# ──────────────────────────────────────────────────────────────────────────────
class Scenario(ExpectationMixin):  # was: class Scenario:
    """
    BDD-like scenario runner with:
      - expectation helpers (exact/contains/regex/schema)
      - runs-per-test & accuracy threshold
      - scenario-level seed override (+ optional per-run randomization)
      - optional raw-output dumping
      - multi-shot control (use/ignore history across steps and runs)
      - rich, colorized, per-run table output
    """

    def __init__(self, name: str, provider):
        self.name = name
        self.provider = provider

        # Multi-shot: default from provider if available, else True
        provider_default_multi = getattr(provider, "default_multi_shot", True)
        self._multi_shot: bool = provider_default_multi  # can be overridden per scenario

        self.system_prompt: Optional[str] = None
        # Seed history provided via .given(); used only when multi-shot is True
        self._seed_history: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "content": "..."}]

        # Step builder state
        self._steps: List[Dict[str, Any]] = []  # each: {"prompt": str, "kind": ..., ...}
        self._prompt_text: Optional[str] = None
        self._random_prompt_choices: Optional[List[str]] = None  # added

        # Expectations (for the *current* step being built)
        self._expect_kind: Optional[str] = None
        self._expect_value: Any = None
        self._expect_regex: Optional[re.Pattern] = None
        self._expect_model: Optional[Type[BaseModel]] = None
        self._expect_model_values: Dict[str, Any] = {}
        self._expect_coerce_json: bool = True

        # Runs & accuracy
        self._runs_per_test: int = DEFAULT_RUNS
        self._accuracy_threshold: Optional[float] = DEFAULT_ACC_THRESHOLD

        # Seed controls
        self._seed: Optional[int] = None                 # fixed seed override for this scenario (None => use provider default)
        self._randomize_seed: bool = False               # when True, each run uses a random seed

        # Debug / artifacts
        self._last_outputs: List[str] = []

        # Raw output dumping controls (inherit suite defaults)
        self._raw_dump_mode: str = RAW_DUMP_MODE  # "never" | "fail" | "always"
        self._raw_dump_dir: str = RAW_DUMP_DIR  # "" => console only
        self._raw_file_format: str = "txt"  # "txt" | "json"

    # ── BDD-ish API ────────────────────────────────────────────────────────────
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        return self

    def given(self, content: str):
        """Seed initial history for each run (only used when multi_shot==True)."""
        self._seed_history.append({"role": "user", "content": content})
        return self

    def when(self, content: str):
        return self.prompt(content)

    def prompt(self, prompt_text: str):
        """Start a new step. If a previous step was in progress, finalize it."""
        self._finalize_pending_step()
        self._prompt_text = prompt_text
        return self

    def random_prompt(self, prompts: List[str]):
        """
        Define a step whose user prompt is randomly chosen from the provided list
        for each run (and each time the step is executed).
        Chain expectation methods after this just like with prompt().
        """
        self._finalize_pending_step()
        if not prompts:
            raise ValueError("random_prompt requires at least one prompt string")
        self._random_prompt_choices = list(prompts)
        return self

    # ── Controls: seed, runs, accuracy, multi_shot ─────────────────────────────
    def seed(self, value: Optional[int]):
        """Scenario-level seed (overrides provider seed). Pass None to clear."""
        self._seed = value if value is None else int(value)
        return self

    def randomize_seed(self, enabled: bool = True):
        """
        If enabled, each run will use a different random seed.
        This does not change the provider default; it only affects this scenario's runs.
        """
        self._randomize_seed = bool(enabled)
        return self

    def runs(self, n: int):
        self._runs_per_test = max(1, int(n))
        return self

    def require_accuracy(self, threshold: float):
        self._accuracy_threshold = float(threshold)
        return self

    def multi_shot(self, value: bool = True):
        """
        Control whether each run chains steps with history (True) or treats each call as fresh (False).
        - True  (default): per-run, all steps are executed with accumulated history.
        - False: only the first step is executed per run; no history is passed.
        """
        self._multi_shot = bool(value)
        return self

    # ── Raw output controls ────────────────────────────────────────────────────
    def dump_raw(self, mode: str = "fail", to_dir: Optional[str] = None, file_format: str = "txt"):
        """
        Dump raw LLM outputs.
          mode: "never" | "fail" | "always"
          to_dir: directory to write files; if omitted/empty, print to console
          file_format: "txt" | "json"
        """
        mode = (mode or "fail").strip().lower()
        assert mode in ("never", "fail", "always"), "dump_raw mode must be 'never' | 'fail' | 'always'"
        self._raw_dump_mode = mode
        if to_dir is not None:
            self._raw_dump_dir = to_dir
        self._raw_file_format = "json" if str(file_format).lower() == "json" else "txt"
        return self

    # ── Internals: step building & validation ──────────────────────────────────
    def _finalize_pending_step(self):
        """Push the current in-progress step into the steps list."""
        if self._prompt_text is None and self._random_prompt_choices is None:
            return
        step = {
            "prompt": self._prompt_text,
            "prompt_choices": list(self._random_prompt_choices) if self._random_prompt_choices else None,  # added
            "kind": self._expect_kind,
            "value": self._expect_value,
            "regex": self._expect_regex,
            "model": self._expect_model,
            "model_values": dict(self._expect_model_values),
            "coerce_json": self._expect_coerce_json,
        }
        self._steps.append(step)

        # reset current step builder state
        self._prompt_text = None
        self._random_prompt_choices = None  # added
        self._expect_kind = None
        self._expect_value = None
        self._expect_regex = None
        self._expect_model = None
        self._expect_model_values = {}
        self._expect_coerce_json = True

    def _write_raw(self, run_idx: int, output: str):
        """Write raw output to file or console."""
        if self._raw_dump_dir:
            os.makedirs(self._raw_dump_dir, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.name)[:80]
            ext = "json" if self._raw_file_format == "json" else "txt"
            path = os.path.join(self._raw_dump_dir, f"{safe_name}__run{run_idx}.{ext}")
            if self._raw_file_format == "json":
                blob = {"name": self.name, "run": run_idx, "output": output}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(blob, f, ensure_ascii=False, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(output)
            return

        # Console-only dump (compact, with a rule)
        console.rule(f"[dim]RAW OUTPUT[/dim] • {self.name} • run {run_idx}")
        console.print(_truncate(output, 4000))

    def _validate_schema_with_spec(self, output: str, model: Type[BaseModel], model_values: Dict[str, Any], coerce_json: bool) -> Tuple[bool, List[str]]:
        raw = output
        if coerce_json:
            maybe = _extract_json(output)
            if maybe:
                raw = maybe
        try:
            data = json.loads(raw)
        except Exception:
            return False, ["Expected: Valid JSON matching schema", f"Got:      {output}"]
        try:
            parsed = model.model_validate(data)  # pydantic v2
        except ValidationError as ve:
            # Extract missing field names (pydantic v2 error types: 'missing', 'field_required')
            missing_fields: List[str] = []
            for err in ve.errors():
                if err.get("type") in ("missing", "missing_field", "field_required"):
                    loc = err.get("loc")
                    if isinstance(loc, (list, tuple)):
                        missing_fields.append(".".join(str(p) for p in loc))
                    elif loc:
                        missing_fields.append(str(loc))
            lines = ["Expected: JSON matching schema"]
            if missing_fields:
                lines.append("Missing fields: " + ", ".join(sorted(set(missing_fields))))
            lines.append(json.dumps(data, ensure_ascii=False, indent=2))
            lines.append(str(ve))
            return False, lines

        mismatches: List[str] = []
        for k, expected_val in model_values.items():
            got_val = getattr(parsed, k, None)
            if got_val != expected_val:
                mismatches.append(f"Field '{k}': Expected: {expected_val} | Got: {got_val}")
        if mismatches:
            return False, mismatches + [
                "Got (parsed):",
                json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2),
            ]
        return True, []

    def _eval_steps(
        self,
        run_history: List[Dict[str, str]],
        steps: List[Dict[str, Any]],
        multi_shot: bool,
        seed_for_run: Optional[int],
    ) -> Tuple[bool, str, List[str], int]:
        """
        Evaluate all steps for a single run.
        Returns: (passed_all, output_of_failed_or_last, fail_lines, failed_step_index_or_-1)
        """
        last_output = ""
        for idx, spec in enumerate(steps, start=1):
            prompt = spec["prompt"]
            choices = spec.get("prompt_choices")  # added
            if choices:  # added
                prompt = random.choice(choices)
            kind = spec["kind"]
            value = spec["value"]
            regex = spec["regex"]
            model = spec["model"]
            model_values = spec["model_values"]
            coerce_json = spec["coerce_json"]

            # Call provider with current run history and per-run seed
            out = self.provider.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                history=run_history if multi_shot else None,
                seed=seed_for_run,
            )
            last_output = out

            # Evaluate this step
            if kind is None:
                ok = True
                fail_lines: List[str] = []
            elif kind == "exact":
                exp = str(value)
                ok = out.strip() == exp
                fail_lines = [] if ok else [f"Expected: {exp}", f"Got:      {out}"]
            elif kind == "contains":
                exp = str(value)
                ok = exp in out
                fail_lines = [] if ok else [f"Expected: Output to contain: {exp}", f"Got:      {out}"]
            elif kind == "not_equal":
                exp = str(value)
                ok = out.strip() != exp
                fail_lines = [] if ok else [f"Did not expect exactly: {exp}", f"But got: {out}"]
            elif kind == "not_contains":
                exp = str(value)
                ok = exp not in out
                fail_lines = [] if ok else [f"Did not expect substring: {exp}", f"But got: {out}"]
            elif kind == "regex":
                ok = bool(regex and regex.search(out))
                fail_lines = [] if ok else [f"Expected (regex): {regex.pattern if regex else '<none>'}", f"Got: {out}"]
            elif kind == "schema":
                ok, fail_lines = self._validate_schema_with_spec(out, model, model_values, coerce_json)
            else:
                ok = False
                fail_lines = [f"Unknown expectation kind: {kind}"]

            # Update history only if multi-shot
            if multi_shot:
                run_history.append({"role": "user", "content": prompt})
                run_history.append({"role": "assistant", "content": out})

            if not ok:
                return False, out, fail_lines, idx  # early exit on first failed step

        return True, last_output, [], -1

    # ── Runner (pretty, colorized) ─────────────────────────────────────────────
    def run(self):
        name = self.name
        try:
            # Ensure we have at least one finalized step
            self._finalize_pending_step()
            if not self._steps:
                raise ValueError("No steps defined. Use .prompt(...).expect_*(...) before .run().")

            effective_multi = self._multi_shot

            # Header panel
            header = Text()
            header.append("Scenario: ", style="bold cyan")
            header.append(name, style="bold white")
            header.append("\nMode: ", style="cyan")
            header.append("multi-shot" if effective_multi else "single-shot", style="white")
            if self.system_prompt:
                header.append("\nSystem: ", style="cyan")
                header.append(_truncate(self.system_prompt, 120), style="white")

            # If single-shot but multiple steps exist, we warn and only use the first step.
            steps_to_use = self._steps
            if not effective_multi and len(self._steps) > 1:
                header.append(
                    f"\n[warning]Note:[/warning] single-shot is enabled; only the first of {len(self._steps)} steps will execute.",
                    style="yellow",
                )
                steps_to_use = [self._steps[0]]

            # Show a concise prompt preview (first step)
            header.append("\nFirst prompt: ", style="cyan")
            first_step = steps_to_use[0]  # added
            if first_step.get("prompt_choices"):  # added
                header.append(f"<random choice of {len(first_step['prompt_choices'])}>", style="white")
            else:  # existing behavior
                header.append(_truncate(first_step['prompt'], 120), style="white")

            # Seed info
            if self._randomize_seed:
                header.append("\nSeed: ", style="cyan")
                header.append("random per-run", style="white")
            elif self._seed is not None:
                header.append("\nSeed: ", style="cyan")
                header.append(str(self._seed), style="white")

            console.print(Panel(header, box=box.ROUNDED, border_style="cyan", padding=(1, 2)))

            # Per-run table
            table = Table(
                show_header=True,
                header_style="bold white",
                box=box.SIMPLE_HEAVY,
                show_lines=False,
                pad_edge=False,
                expand=True,
            )
            table.add_column("Run", justify="right", width=4)
            table.add_column("Result", width=8)
            table.add_column("Seed", justify="right", width=10)
            table.add_column("Details", overflow="fold")

            passes = 0
            self._last_outputs.clear()
            fail_examples: List[str] = []

            for i in range(1, self._runs_per_test + 1):
                # Per-run seed
                if self._randomize_seed:
                    run_seed: Optional[int] = random.randint(0, 2**31 - 1)
                else:
                    run_seed = self._seed  # could still be None -> provider default

                # Fresh run history each iteration
                run_history = list(self._seed_history) if effective_multi else []
                passed, out, fail_lines, fail_step = self._eval_steps(
                    run_history, steps_to_use, effective_multi, run_seed
                )
                self._last_outputs.append(out)

                # Raw dump policy
                if self._raw_dump_mode == "always" or (self._raw_dump_mode == "fail" and not passed):
                    self._write_raw(i, out)

                badge = _result_badge(passed)
                seed_text = str(run_seed) if run_seed is not None else "—"

                if passed:
                    details = Text(f"OK ({len(steps_to_use)} step{'s' if len(steps_to_use)!=1 else ''})", style="green")
                    passes += 1
                else:
                    first_line = fail_lines[0] if fail_lines else "(mismatch)"
                    prefix = f"Step {fail_step} failed: "
                    # Previously truncated; now show full line without ellipsis
                    details = Text(prefix + first_line, style="red")
                    # Store full output (no truncation)
                    fail_examples.append(f"Step {fail_step}: {first_line}\n{out}")
                table.add_row(str(i), badge, seed_text, details)

            console.print(table)

            # Summary line with accuracy bar
            acc = passes / self._runs_per_test
            summary = Text.assemble(
                ("Summary: ", "cyan"),
                (f"{passes}/{self._runs_per_test} passes  ", "bold"),
            )
            summary_panel = Panel(
                _accuracy_bar(acc) + _threshold_badge(self._accuracy_threshold, acc),
                title=summary,
                title_align="left",
                border_style="cyan" if passes == self._runs_per_test else "red",
                box=box.ROUNDED,
                padding=(1, 2),
            )
            console.print(summary_panel)

            # Threshold enforcement
            failed_by_threshold = self._accuracy_threshold is not None and acc < self._accuracy_threshold
            if failed_by_threshold:
                console.print(
                    Panel(
                        Text("FAIL: accuracy below required threshold", style="bold red"),
                        border_style="red",
                        box=box.SQUARE,
                    )
                )
                if fail_examples:
                    console.print(Text("Examples of failures:", style="bold red"))
                    for ex in fail_examples[:3]:
                        console.print(f"- {ex}")

            # Classic PASS/FAIL footer with examples if any failed
            if passes == self._runs_per_test and not failed_by_threshold:
                console.print(Text("✔ All runs passed.", style="bold green"))
            else:
                console.print(Text("✖ Some runs failed.", style="bold red"))
                if fail_examples:
                    console.print(Text("Examples of failures:", style="bold red"))
                    for ex in fail_examples[:3]:
                        console.print(f"- {ex}")

            # Record in global results registry
            RESULTS.append({
                "name": name,
                "runs": self._runs_per_test,
                "passes": passes,
                "accuracy": acc,
                "failed": (passes < self._runs_per_test) or failed_by_threshold,
            })

            # end of scenario block separator
            console.rule(characters="─", style="dim")

        except Exception:
            console.print(Panel(Text(f"FAIL: {name} (exception)", style="bold white on red"), box=box.HEAVY))
            traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Optional suite summary printer (call from run_tests.py after all scenarios)
# ──────────────────────────────────────────────────────────────────────────────
def print_summary():
    """Render a bottom-of-run summary table using global RESULTS."""
    table = Table(title="LLM Test Suite Summary", header_style="bold white", show_lines=True, box=box.SIMPLE_HEAVY)
    table.add_column("Scenario", style="bold cyan")
    table.add_column("Passes", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Accuracy", justify="right", style="magenta")
    table.add_column("Status", style="bold")

    passed_total = 0
    failed_total = 0

    for r in RESULTS:
        status_ok = not r["failed"]
        status = "[green]PASS[/green]" if status_ok else "[red]FAIL[/red]"
        if status_ok:
            passed_total += 1
        else:
            failed_total += 1

        table.add_row(
            r["name"],
            str(r["passes"]),
            str(r["runs"]),
            f"{r['accuracy']*100:.1f}%",
            status,
        )

    console.print(table)
    console.print(
        Panel(
            Text.assemble(
                ("Total: ", "cyan"),
                (f"{passed_total+failed_total} ", "bold white"),
                (f"✔ {passed_total} passed ", "green"),
                (f"✖ {failed_total} failed", "red"),
            ),
            border_style="cyan",
        )
    )