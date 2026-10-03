"""LLM loop plumbing with a fake client (no API access needed)."""
import json
from types import SimpleNamespace as NS

from flybrainlab.agent import llm


class FakeTools:
    def __init__(self):
        self.calls, self.reports, self.trials_used, self.transcript = [], [], 0, []
        self.dataset = "toy"

    def call_json(self, name, args):
        self.calls.append((name, args))
        if name == "submit_report":
            self.reports.append("report.md")
            return json.dumps({"accepted": True}), False
        return json.dumps({"ok": True, "echo": args}), False


class FakeClient:
    def __init__(self):
        self.requests = []
        script = [
            [NS(type="text", text="plan"), NS(type="tool_use", id="t1", name="list_neuron_groups", input={"query": "GF"}),
             NS(type="tool_use", id="t2", name="describe_scenarios", input={})],
            [NS(type="tool_use", id="t3", name="submit_report", input={"title": "x"})],
        ]
        self._it = iter(script)
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        content = next(self._it)
        return NS(content=content, stop_reason="tool_use", stop_details=None)


def test_agent_loop_threads_tool_results(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    client, tools = FakeClient(), FakeTools()
    log = llm.run_agent("question?", client=client, tools=tools, verbose=False)
    assert [c[0] for c in tools.calls] == ["list_neuron_groups", "describe_scenarios", "submit_report"]
    second = client.requests[1]
    assert second["model"] == llm.DEFAULT_MODEL and second["thinking"] == {"type": "adaptive"}
    results = second["messages"][-1]["content"]  # both tool results in ONE user message
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert log["reports"] == ["report.md"]
