"""Render docs/results.md from docs/validation/*.json (numbers copied, never typed by hand)."""
from __future__ import annotations

import json

from flybrainlab.experiments.store import RunStore
from flybrainlab.paths import ROOT

V = ROOT / "docs" / "validation"


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def study_table(name: str, st: dict, store: RunStore) -> list[str]:
    runs = st["runs"]
    metrics = [m["metric"] for m in next(iter(st["comparisons"].values()))["metrics"]]
    cols = [(r["label"], r["run_id"]) for r in runs] + [("intact, shuffled", runs[0]["control_run_id"])]
    lines = ["| Metrik | " + " | ".join(c[0] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in metrics:
        cells = []
        for _, rid in cols:
            s = store.summary(rid)["metrics"].get(m, {})
            cells.append(f"{fmt(s.get('mean'))} ± {fmt(s.get('std'))} (n={s.get('n', 0)})" if s.get("n") else "– (n=0)")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "Tests (gegen *intact*; Fisher exakt für binäre Metriken, sonst Welch-t; p nach Holm über die Metriken):", ""]
    for key, comp in st["comparisons"].items():
        parts = [f"{r['metric']}: p={fmt(r.get('p_holm', r.get('p_value')))}" for r in comp["metrics"] if "p_value" in r]
        lines.append(f"- intact {key.replace('vs_', 'vs. ')}: " + ("; ".join(parts) if parts else "kein Test möglich"))
    lines += ["", "Runs: " + ", ".join(f"`{rid}`" for _, rid in cols), "", f"![{name}](validation/ref_{name}.png)", ""]
    return lines


def main():
    store = RunStore()
    out = ["# Ergebnisse der Referenz-Experimente", "",
           "Automatisch erzeugt von `scripts/make_results_md.py` aus `docs/validation/*.json` und den Runs in `runs/`. Kopien aller hier genannten Runs (Config, Seeds, Metriken pro Trial, Trajektorien) liegen versioniert unter `runs/examples/`.",
           "Werte: Mittelwert ± SD über Seeds (n = gültige Trials). Datensatz: MaleCNS v1.0, Körper `simple`.",
           "Alle Aussagen gelten für **Konnektom + engineered Encoder + handgesetzte Ausleseschicht**.", ""]
    m1 = json.loads((V / "m1_results.json").read_text())
    fw = m1["flywire_v783"]
    out += ["## M1: Gehirn allein (Shiu-Befund)", "",
            "| | MN9-Rate [Hz] | aktive Neuronen |", "|---|---|---|",
            f"| FlyWire v783, Brian2-Referenz (n={fw['n_trials_brian2']}) | {fmt(fw['mn9_rate_brian2_hz'][0])} ± {fmt(fw['mn9_rate_brian2_hz'][1])} | {fw['n_active_brian2']} |",
            f"| FlyWire v783, diese Engine (n={fw['n_trials_ours']}) | {fmt(fw['mn9_rate_ours_hz'][0])} ± {fmt(fw['mn9_rate_ours_hz'][1])} | {fw['n_active_ours']} |",
            f"| FlyWire v783, Shuffle | {fmt(fw['mn9_rate_shuffled_hz'][0])} | |", "",
            f"Korrelation der Raten aller {fw['n_responsive_neurons']} antwortenden Neuronen (Engine vs. Brian2): "
            f"r = {fw['rate_correlation_responsive_neurons']:.4f}.", ""]
    mc = m1["malecns_v1.0"]
    out += ["| MaleCNS v1.0 (w_syn 0,15 mV, n=8) | intakt [Hz] | Shuffle [Hz] |", "|---|---|---|"]
    for d in mc.values():
        for t, v in d.items():
            if isinstance(v, dict):
                out.append(f"| {d['source']} @ {fmt(d['rate_hz'])} Hz → {t} | {fmt(v['intact_hz'][0])} ± {fmt(v['intact_hz'][1])} | {fmt(v['shuffled_hz'][0])} |")
    out += ["", "![M1](validation/m1_validation.png)", ""]
    ref = json.loads((V / "reference_runs.json").read_text())
    titles = {"looming": "M3: Looming-Flucht", "spider": "M4: Spinne", "zerog": "M6: Schwerelosigkeit (Luft bleibt)",
              "lightchoice": "M7: Lichtwahl (T-Labyrinth)", "foraging": "M7: Futtersuche (Duft)"}
    for name in ("looming", "spider", "zerog", "lightchoice", "foraging"):
        if name in ref:
            out += [f"## {titles[name]}", ""] + study_table(name, ref[name], store)
    esc = V / "escape_readout_malecns_v1.0.json"
    if esc.exists():
        e = json.loads(esc.read_text())
        out += ["## Trainierte Ausleseschicht: Sprungrichtung", "",
                f"Ridge-Regression von {e['n_descending']} Descending Neurons (gefilterte Rate beim Giant-Fiber-Trigger) auf die "
                f"Fluchtrichtung, {e['n_azimuths']} Azimute × {e['seeds']} Seeds, Leave-one-azimuth-out-Kreuzvalidierung.", "",
                "| Konnektom | aktive DNs | Abheben ausgelöst | mittl. Winkelfehler (CV) | Median |", "|---|---|---|---|---|"]
        for k in ("intact", "shuffled"):
            v = e[k]
            out.append(f"| {k} | {v['n_active_dn']} | {fmt(v['takeoff_fraction'])} | {fmt(v['cv_mean_abs_error_deg'])}° | {fmt(v['cv_median_abs_error_deg'])}° |")
        out += [f"| Zufall | | | {fmt(e['chance_mean_abs_error_deg'])}° | |", ""]
        tr = V / "trained_readout_run.json"
        if tr.exists():
            from flybrainlab.experiments.stats import compare_runs

            t = json.loads(tr.read_text())
            base = ref["looming"]["runs"][0]["run_id"]
            c = compare_runs(base, t["run_id"], ["escape_dir_error_deg", "takeoff"], store=store)["metrics"][0]
            out += ["Im geschlossenen Loop (Looming, gleiche 12 Seeds, Azimute nicht im Training):", "",
                    "| Ausleseschicht | escape_dir_error_deg |", "|---|---|",
                    f"| handgesetzt (DNp02/DNp11 L−R) | {fmt(c['mean_a'])} ± {fmt(c['std_a'])} (n={c['n_a']}) |",
                    f"| trainiert (Ridge, alle DNs) | {fmt(c['mean_b'])} ± {fmt(c['std_b'])} (n={c['n_b']}) |", "",
                    f"Welch-t p = {fmt(c['p_value'])}. Runs: `{base}`, `{t['run_id']}`. Berichte dazu müssen "
                    "\"Konnektom + trainierte Ausleseschicht\" sagen.", ""]
    (ROOT / "docs" / "results.md").write_text("\n".join(out))
    print("wrote docs/results.md")


if __name__ == "__main__":
    main()
