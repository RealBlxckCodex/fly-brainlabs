"""LLM experimenter: Claude plans experiments and reports, but only through tools.

Manual tool-use loop (Anthropic Python SDK). The model can only act through the tools
in tools.py; all numbers come from computed metrics and the final report is validated.
"""
from __future__ import annotations

import json
from datetime import datetime

from ..paths import RUNS
from .tools import TOOL_SCHEMAS, ExperimentTools

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are the automated experimenter of Fly Brain Lab. A complete Drosophila connectome runs as a
leaky integrate-and-fire network inside a closed loop with a simulated fly body and environment.
Your job: turn the user's research question into a small series of controlled simulation experiments,
then write a report.

How you work
- You cannot observe the simulation directly. Everything you know comes from tool results. Report only
  numbers returned by get_metrics / compare / run_experiment; never estimate or round up "by feel".
- Every experiment runs several seeds (n_trials >= 6 unless the budget forbids it) and comes with an
  automatic shuffled-connectome control. Compare intervention vs. intact baseline AND intact vs. shuffle:
  a behaviour that survives shuffling does not depend on the real wiring.
- Use the same seed and n_trials for conditions you will compare, so the scenario randomisation matches.
- Use list_neuron_groups before lesioning to check that a group exists and how many neurons it has.
- Keep the trial budget in mind; prefer a few decisive comparisons over many small ones.
- Finish by calling submit_report. It states: hypothesis, intervention, metrics with spread, control, model
  limitations, and a conclusion phrased as a hypothesis about the model, not a claim about real flies.
  The report is machine-checked: each finding cites run_id + metric + statistic, and any number in the
  text must match a tool result. If it is rejected, fix the listed problems and resubmit.
- The readout from descending neurons to the body is engineered (not connectome). Say so."""


def run_agent(question: str, dataset: str | None = None, model: str = DEFAULT_MODEL, trial_budget: int = 120,
              max_turns: int = 40, effort: str = "high", workers: int | None = None, verbose: bool = True,
              client=None, tools: ExperimentTools | None = None) -> dict:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    tools = tools or ExperimentTools(dataset=dataset, trial_budget=trial_budget, workers=workers, verbose=verbose)
    messages: list = [{"role": "user", "content": question}]
    log = {"question": question, "model": model, "dataset": tools.dataset, "started": datetime.now().isoformat(),
           "turns": []}
    final_text = ""
    for turn in range(max_turns):
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        messages.append({"role": "assistant", "content": response.content})
        texts = [b.text for b in response.content if b.type == "text"]
        if texts and verbose:
            print("\n[claude] " + "\n".join(texts), flush=True)
        final_text = "\n".join(texts) or final_text
        log["turns"].append({"stop_reason": response.stop_reason, "text": texts,
                             "tool_calls": [{"name": b.name, "input": b.input} for b in response.content
                                            if b.type == "tool_use"]})
        if response.stop_reason == "refusal":
            log["refusal"] = getattr(response.stop_details, "category", None) if response.stop_details else None
            break
        if response.stop_reason == "max_tokens":
            messages.append({"role": "user", "content": "Your last reply was cut off; continue more concisely."})
            continue
        if response.stop_reason != "tool_use":
            if tools.reports:
                break
            messages.append({"role": "user", "content": "Please finish by calling submit_report."})
            continue
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"[tool] {block.name} {json.dumps(block.input)[:300]}", flush=True)
            content, is_error = tools.call_json(block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
        if tools.reports and all(not r["is_error"] for r in results) and any(
                b.type == "tool_use" and b.name == "submit_report" for b in response.content):
            break
    log.update(finished=datetime.now().isoformat(), reports=tools.reports, trials_used=tools.trials_used,
               final_text=final_text, transcript=tools.transcript)
    out = RUNS / "agent_sessions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(log, indent=2, default=str))
    log["session_log"] = str(path)
    return log
