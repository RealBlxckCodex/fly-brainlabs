"""Report validation and rendering.

Rule enforcement:
1. Every finding cites (run_id, metric, statistic, value); the value is checked against
   the stored summary.json (tolerance: rounding to the written precision, or 1e-6 rel).
2. Every number written in free text must be traceable to an earlier tool result
   (or its percentage form). Small integers 0..10 are allowed (counts, list numbering).
3. Reports must reference at least one control run.
4. The fixed model-limitations block is appended to every report.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .. import provenance
from ..experiments.store import RunStore

STANDARD_LIMITATIONS = """\
- Model = static connectome (synapse counts, predicted transmitter sign) + leaky integrate-and-fire dynamics
  (Shiu et al. 2024). No plasticity, no neuromodulation, no gap junctions, no cell-type-specific membrane properties.
- w_syn is a free parameter (calibrated per dataset: 0.275 mV FlyWire, 0.15 mV MaleCNS).
- Sensory encoders (looming, photoreceptors, odour, Johnston's organ) are engineered mappings onto real neuron
  groups, not models of the periphery.
- Motor side: descending-neuron rates -> hand-set readout -> motor primitives (walk/turn/jump/flight wrench).
  No joint-level control; wing aerodynamics are abstracted. Report as "connectome + engineered readout".
- Results are hypotheses about the wiring diagram, not evidence about real flies."""

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


class ReportError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _collect_numbers(obj, out: set):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect_numbers(v, out)
            for m in NUM_RE.findall(str(k)):
                out.add(float(m))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            try:
                out.add(float(m))
            except ValueError:
                pass


def _decimals(tok: str) -> int:
    tok = tok.lower()
    if "e" in tok:
        return 12
    return len(tok.split(".")[1]) if "." in tok else 0


def number_is_supported(tok: str, seen: set[float]) -> bool:
    x = float(tok)
    if x.is_integer() and 0 <= abs(x) <= 10:
        return True
    tol = 0.5 * 10 ** (-_decimals(tok)) + 1e-12
    for v in seen:
        for cand in (v, v * 100.0, abs(v), abs(v) * 100.0):
            if abs(abs(x) - cand) <= tol or (x == 0 and abs(cand) <= tol):
                return True
    return False


def _stat(summary: dict, metric: str, statistic: str):
    m = summary["metrics"].get(metric)
    if m is None:
        return None
    if statistic == "ci95_low":
        return m.get("ci95", [None, None])[0]
    if statistic == "ci95_high":
        return m.get("ci95", [None, None])[1]
    return m.get(statistic)


def validate_and_write_report(report: dict, transcript: list[dict], store: RunStore) -> Path:
    problems = []
    seen: set[float] = set()
    for entry in transcript:
        _collect_numbers(entry.get("result"), seen)
        _collect_numbers(entry.get("input"), seen)
    known_runs = {r["run_id"] for r in store.list()}
    for rid in report.get("runs", []) + report.get("controls", []):
        if rid not in known_runs:
            problems.append(f"unknown run_id {rid}")
    if not report.get("controls"):
        problems.append("no control run listed (shuffle control is mandatory)")
    for k, f in enumerate(report.get("findings", [])):
        try:
            v = _stat(store.summary(f["run_id"]), f["metric"], f["statistic"])
        except KeyError:
            problems.append(f"finding {k}: unknown run_id {f['run_id']}")
            continue
        if v is None:
            problems.append(f"finding {k}: metric/statistic {f['metric']}.{f['statistic']} not in run {f['run_id']}")
        elif abs(v - f["value"]) > max(1e-6 * abs(v), 0.5 * 10 ** -3) and abs(v - f["value"]) > 0.005 * abs(v):
            problems.append(f"finding {k}: value {f['value']} != stored {f['metric']}.{f['statistic']} = {v:.6g}")
    for c in report.get("comparisons", []):
        try:
            from ..experiments.stats import compare_runs

            cmp = compare_runs(c["run_a"], c["run_b"], [c["metric"]], store=store)["metrics"]
        except KeyError as e:
            problems.append(f"comparison: {e}")
            continue
        if not cmp or "p_value" not in cmp[0]:
            problems.append(f"comparison {c['metric']}: no test available")
        elif abs(cmp[0]["p_value"] - c["p_value"]) > max(0.005 * cmp[0]["p_value"], 5e-4):
            problems.append(f"comparison {c['metric']}: p_value {c['p_value']} != computed {cmp[0]['p_value']:.4g}")
    text_fields = ["title", "hypothesis", "intervention", "spread", "control_assessment", "limitations", "conclusion"]
    texts = [(f, report.get(f, "")) for f in text_fields]
    texts += [(f"findings[{k}].statement", f["statement"]) for k, f in enumerate(report.get("findings", []))]
    texts += [(f"comparisons[{k}].interpretation", c["interpretation"]) for k, c in enumerate(report.get("comparisons", []))]
    for field, txt in texts:
        clean = re.sub(r"\b\d{8}-\d{6}_[\w]+", "", txt or "")  # run ids
        clean = re.sub(r"\b[A-Za-z]+\d+[A-Za-z0-9_-]*", "", clean)  # names like DNa02, LC4, R1-R6
        for tok in NUM_RE.findall(clean):
            if not number_is_supported(tok, seen):
                problems.append(f"{field}: number '{tok}' does not appear in any tool result")
    if problems:
        raise ReportError(problems)
    return write_report(report, store)


def write_report(report: dict, store: RunStore) -> Path:
    out = store.root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", report["title"].lower()).strip("-")[:50]
    stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{slug}"
    git = provenance.git_commit()
    lines = [f"# {report['title']}", "",
             f"*Generated {provenance.utc_now()} | git {git['commit'][:10]}{' (dirty)' if git['dirty'] else ''}*", "",
             "## Hypothesis", report["hypothesis"], "", "## Intervention", report["intervention"], "",
             "## Runs", *[f"- `{r}` - {store.config(r).get('label')}" for r in report["runs"]], "",
             "## Controls", *[f"- `{r}` - {store.config(r).get('label')}" for r in report["controls"]], "",
             "## Findings (values verified against stored metrics)", ""]
    lines += ["| statement | run | metric | statistic | value |", "|---|---|---|---|---|"]
    for f in report["findings"]:
        lines.append(f"| {f['statement']} | `{f['run_id']}` | {f['metric']} | {f['statistic']} | {f['value']:.4g} |")
    if report.get("comparisons"):
        lines += ["", "## Statistical comparisons (recomputed)", "",
                  "| run a | run b | metric | p | interpretation |", "|---|---|---|---|---|"]
        for c in report["comparisons"]:
            lines.append(f"| `{c['run_a']}` | `{c['run_b']}` | {c['metric']} | {c['p_value']:.3g} | {c['interpretation']} |")
    lines += ["", "## Spread across seeds", report["spread"], "", "## Controls assessment", report["control_assessment"],
              "", "## Limitations", report["limitations"], "", "### Standard model limitations (always included)",
              STANDARD_LIMITATIONS, "", "## Conclusion (hypothesis)", report["conclusion"], ""]
    md = out / f"{stem}.md"
    md.write_text("\n".join(lines))
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2))
    return md
