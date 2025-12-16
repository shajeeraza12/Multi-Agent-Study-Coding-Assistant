import types
import pytest

import agents

# Test that _call_llm handles different LLM APIs

def test_call_llm_invoke():
    class Dummy:
        def invoke(self, arg):
            return "invoked" + str(arg)

    d = Dummy()
    assert agents._call_llm(d, "x") == "invokedx"


def test_call_llm_run():
    class Dummy:
        def run(self, arg):
            return "run" + str(arg)

    d = Dummy()
    assert agents._call_llm(d, "x") == "runx"


def test_call_llm_callable():
    def func(x):
        return "call" + str(x)

    assert agents._call_llm(func, "x") == "callx"


def test_researcher_with_callable_tavily_tool(monkeypatch):
    # Replace tavily_tool with a simple callable
    def fake_tavily_call(kwargs):
        return {"results": [{"title": "Test", "url": "http://example.com", "content": "Example"}]}

    monkeypatch.setattr(agents, "tavily_tool", fake_tavily_call)
    researcher = agents.create_researcher_agent()
    out = researcher({"input": "test query"})
    assert "output" in out and isinstance(out["output"], str)
