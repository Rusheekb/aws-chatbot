"""Tests for tools.py: the dispatcher turns every outcome into a valid toolResult."""
import tools


def call(name, tool_input):
    return {"toolUseId": "id-1", "name": name, "input": tool_input}


def test_successful_tool_returns_json(monkeypatch):
    monkeypatch.setitem(tools.TOOL_FUNCTIONS, "echo", lambda text: {"echo": text})

    result = tools.run_tool(call("echo", {"text": "hi"}))["toolResult"]

    assert result == {"toolUseId": "id-1", "content": [{"json": {"echo": "hi"}}], "status": "success"}


def test_unknown_tool_is_an_error_not_a_crash():
    result = tools.run_tool(call("does_not_exist", {}))["toolResult"]
    assert result["status"] == "error"
    assert "No tool named does_not_exist" in result["content"][0]["text"]


def test_wrong_arguments_are_reported_to_the_model(monkeypatch):
    monkeypatch.setitem(tools.TOOL_FUNCTIONS, "echo", lambda text: {"echo": text})
    result = tools.run_tool(call("echo", {"wrong": "x"}))["toolResult"]
    assert result["status"] == "error"


def test_every_spec_has_a_matching_function():
    spec_names = {tool["toolSpec"]["name"] for tool in tools.TOOL_CONFIG["tools"]}
    assert spec_names == set(tools.TOOL_FUNCTIONS)
