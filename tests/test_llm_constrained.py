"""Constrained decoding and the bounded agent loop, against a fake provider.

Two properties, both testable offline:

  * the schema reaches the provider as a forced tool call, not only as a
    cache key or a post-hoc parse;
  * the agent loop issues model calls and executes the tools it asks for.

Nothing here talks to AWS.
"""

from __future__ import annotations

import pytest

from _fakes import FakeBedrock, text as _text, tool_use as _tool_use
from kgprov.llm import AgentRun, BedrockClient, SchemaValidationError


# ---------------------------------------------------------------------
#  Fake provider
# ---------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    c = BedrockClient(region="us-east-1",
                      response_log_dir=str(tmp_path / "log"))
    c._client = None
    return c


ENUM_SCHEMA = {
    "type": "object",
    "properties": {"mapped_id": {"type": "string",
                                 "enum": ["D1", "D2", "NO_MATCH"]}},
    "required": ["mapped_id"],
}


# ---------------------------------------------------------------------
#  Cache keys
# ---------------------------------------------------------------------

def test_constrained_and_unconstrained_never_share_a_cache_entry(client):
    """Constrained, unconstrained and agent calls use distinct cache keys.

    Otherwise a cached unconstrained response could be replayed into a run
    that records decode-time enforcement.
    """
    a = client._key("m", "s", "u", ENUM_SCHEMA)
    b = client._key_constrained("m", "s", "u", "submit", ENUM_SCHEMA)
    c = client._key_agent("m", "s", "u", [{"x": 1}])
    assert len({a, b, c}) == 3


def test_constrained_key_depends_on_the_tool_schema(client):
    other = {**ENUM_SCHEMA, "properties": {
        "mapped_id": {"type": "string", "enum": ["D9"]}}}
    assert (client._key_constrained("m", "s", "u", "submit", ENUM_SCHEMA)
            != client._key_constrained("m", "s", "u", "submit", other))


# ---------------------------------------------------------------------
#  Constrained decoding
# ---------------------------------------------------------------------

def test_schema_is_sent_to_the_provider_as_a_forced_tool(client):
    """The schema is sent as a tool in toolConfig, and toolChoice forces it."""
    client._client = FakeBedrock([_tool_use("submit", {"mapped_id": "D2"})])
    client.complete_constrained(
        "model", "sys", "usr", tool_name="submit", tool_schema=ENUM_SCHEMA)

    sent = client._client.requests[0]
    assert "toolConfig" in sent, "schema did not reach the provider"
    assert sent["toolConfig"]["toolChoice"] == {"tool": {"name": "submit"}}
    spec = sent["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["inputSchema"]["json"] == ENUM_SCHEMA
    assert spec["inputSchema"]["json"]["properties"]["mapped_id"]["enum"]


def test_result_is_read_from_the_tool_block_not_scraped_from_text(client):
    client._client = FakeBedrock([_tool_use("submit", {"mapped_id": "D2"})])
    r = client.complete_constrained(
        "model", "sys", "usr", tool_name="submit", tool_schema=ENUM_SCHEMA)
    assert r.parsed == {"mapped_id": "D2"}


def test_missing_tool_block_is_an_error_not_a_silent_degrade(client):
    """With toolChoice forced this is a provider failure. Surface it."""
    client._client = FakeBedrock([_text('{"mapped_id": "D2"}')])
    with pytest.raises(SchemaValidationError):
        client.complete_constrained("model", "sys", "usr",
                                    tool_name="submit",
                                    tool_schema=ENUM_SCHEMA)


def test_constrained_call_is_cached_and_replayed(client):
    client._client = FakeBedrock([_tool_use("submit", {"mapped_id": "D1"})])
    a = client.complete_constrained("model", "sys", "usr",
                                    tool_name="submit",
                                    tool_schema=ENUM_SCHEMA)
    b = client.complete_constrained("model", "sys", "usr",
                                    tool_name="submit",
                                    tool_schema=ENUM_SCHEMA)
    assert a.parsed == b.parsed
    assert b.from_cache and not a.from_cache
    assert len(client._client.requests) == 1


# ---------------------------------------------------------------------
#  The agent loop
# ---------------------------------------------------------------------

SUBMIT_SCHEMA = {"type": "object",
                 "properties": {"final_cui": {"type": "string"}},
                 "required": ["final_cui"]}


def _agent(client, responses, executor=None, **kw):
    client._client = FakeBedrock(responses)
    calls = []

    def default_executor(name, args):
        calls.append((name, args))
        return "Search results: D008687 Hypoxia"

    run = client.run_agent(
        "model", "sys", "usr",
        tools=[BedrockClient.tool_spec("search", "search the web",
                                       {"type": "object"})],
        executor=executor or default_executor,
        submit_tool_name="submit", submit_schema=SUBMIT_SCHEMA, **kw)
    return run, calls


def test_agent_issues_model_calls_and_executes_tools(client):
    """The property that distinguishes an agent from a lookup."""
    run, calls = _agent(client, [
        _tool_use("search", {"query": "hypoxemia"}),
        _text("found it"),
        _tool_use("submit", {"final_cui": "D008687"}, stop="end_turn"),
    ])
    assert isinstance(run, AgentRun)
    assert run.n_tool_calls == 1
    assert calls == [("search", {"query": "hypoxemia"})]
    assert run.submitted == {"final_cui": "D008687"}


def test_the_model_composes_its_own_query(client):
    """Agency lives in the arguments, not in the tool."""
    _, calls = _agent(client, [
        _tool_use("search", {"query": "acute liver insult MeSH"}),
        _text("ok"),
        _tool_use("submit", {"final_cui": "D017114"}, stop="end_turn"),
    ])
    assert calls[0][1]["query"] == "acute liver insult MeSH"


def test_first_turn_does_not_force_a_tool_choice(client):
    """On the first turn the model may decline to call any tool."""
    _agent(client, [
        _text("no search needed"),
        _tool_use("submit", {"final_cui": "UNMAPPED"}, stop="end_turn"),
    ])
    first = client._client.requests[0]
    assert "toolChoice" not in first["toolConfig"]


def test_agent_may_decline_to_search_at_all(client):
    run, calls = _agent(client, [
        _text("nothing to look up"),
        _tool_use("submit", {"final_cui": "UNMAPPED"}, stop="end_turn"),
    ])
    assert run.n_tool_calls == 0
    assert calls == []
    assert run.submitted == {"final_cui": "UNMAPPED"}


def test_budget_is_enforced_by_the_controller(client):
    """The tool-call budget is enforced by the controller, not the prompt."""
    responses = [_tool_use("search", {"query": f"q{i}"}) for i in range(10)]
    responses.append(_tool_use("submit", {"final_cui": "UNMAPPED"},
                               stop="end_turn"))
    run, calls = _agent(client, responses, max_tool_calls=3)
    assert run.n_tool_calls == 3
    assert len(calls) == 3
    assert run.budget_exhausted is True


def test_submission_is_forced_at_the_end(client):
    _agent(client, [
        _text("done"),
        _tool_use("submit", {"final_cui": "D1"}, stop="end_turn"),
    ])
    last = client._client.requests[-1]
    assert last["toolConfig"]["toolChoice"] == {"tool": {"name": "submit"}}


def test_tool_failure_is_fed_back_not_raised(client):
    def boom(name, args):
        raise RuntimeError("registry down")

    run, _ = _agent(client, [
        _tool_use("search", {"query": "x"}),
        _text("recovering"),
        _tool_use("submit", {"final_cui": "UNMAPPED"}, stop="end_turn"),
    ], executor=boom)
    assert run.n_tool_calls == 1
    assert "registry down" in run.tool_calls[0]["result_excerpt"]


def test_tool_result_carries_the_matching_tool_use_id(client):
    _agent(client, [
        _tool_use("search", {"query": "x"}),
        _text("ok"),
        _tool_use("submit", {"final_cui": "D1"}, stop="end_turn"),
    ])
    msgs = client._client.requests[1]["messages"]
    results = [b["toolResult"] for m in msgs for b in m.get("content", [])
               if isinstance(b, dict) and "toolResult" in b]
    assert results and results[0]["toolUseId"] == "tu_search"


def test_token_usage_accumulates_across_every_turn(client):
    run, _ = _agent(client, [
        _tool_use("search", {"query": "x"}, tokens=(100, 20)),
        _text("ok", tokens=(200, 30)),
        _tool_use("submit", {"final_cui": "D1"}, stop="end_turn",
                  tokens=(300, 40)),
    ])
    assert run.input_tokens == 600
    assert run.output_tokens == 90


# ---------------------------------------------------------------------
#  Parallel tool use
# ---------------------------------------------------------------------

def _multi_tool_use(names_inputs, stop="tool_use"):
    """One assistant turn requesting several tool calls at once."""
    return {
        "output": {"message": {"content": [
            {"toolUse": {"toolUseId": f"tu_{i}", "name": n, "input": inp}}
            for i, (n, inp) in enumerate(names_inputs)]}},
        "stopReason": stop,
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


def test_every_tool_use_in_a_turn_gets_a_result(client):
    """Every toolUse block in a turn gets its own toolResult.

    A turn may contain several toolUse blocks. Bedrock requires a
    toolResult for each one in the following message; answering only the
    first fails the next request with "Expected toolResult blocks at
    messages.N.content".
    """
    run, calls = _agent(client, [
        _multi_tool_use([("search", {"query": "a"}), ("search", {"query": "b"})]),
        _text("ok"),
        _tool_use("submit", {"final_cui": "D1"}, stop="end_turn"),
    ])
    followup = client._client.requests[1]["messages"]
    results = [b["toolResult"] for m in followup for b in m.get("content", [])
               if isinstance(b, dict) and "toolResult" in b]
    assert len(results) == 2, "every toolUse needs its own toolResult"
    assert {r["toolUseId"] for r in results} == {"tu_0", "tu_1"}
    assert run.n_tool_calls == 2


def test_budget_reached_mid_turn_still_answers_every_block(client):
    """Declining must be explicit: a dropped block malforms the exchange."""
    run, calls = _agent(client, [
        _multi_tool_use([("search", {"query": "a"}), ("search", {"query": "b"}),
                         ("search", {"query": "c"})]),
        _tool_use("submit", {"final_cui": "UNMAPPED"}, stop="end_turn"),
    ], max_tool_calls=2)
    followup = client._client.requests[1]["messages"]
    results = [b["toolResult"] for m in followup for b in m.get("content", [])
               if isinstance(b, dict) and "toolResult" in b]
    assert len(results) == 3, "all three blocks must be answered"
    assert run.n_tool_calls == 2, "but only two may actually execute"
    assert run.budget_exhausted is True


# ---------------------------------------------------------------------
#  The retry handler must survive the errors it exists to absorb
# ---------------------------------------------------------------------

@pytest.mark.parametrize("resp", [
    None,                       # botocore sets this on some transport faults
    {},                         # present but empty
    {"Error": None},            # present, Error key null
    {"Error": {}},              # Error present, no Code
    "not-a-dict",               # defensive: never assume shape
])
def test_error_code_survives_a_malformed_exception(resp):
    """_error_code returns "" for any malformed `response` attribute.

    botocore sets `response` to None on some transport failures, so
    `hasattr(e, "response")` is not a sufficient guard before `.get`.
    """
    e = Exception("boom")
    e.response = resp
    assert BedrockClient._error_code(e) == ""


def test_error_code_reads_a_well_formed_throttle():
    e = Exception("boom")
    e.response = {"Error": {"Code": "ThrottlingException"}}
    assert BedrockClient._error_code(e) == "ThrottlingException"


def test_a_none_response_error_surfaces_instead_of_masking_itself(client):
    """The original fault reaches the caller, not an AttributeError.

    An unrecognised error is not retried; it is raised as it came.
    """
    class Flaky:
        def converse(self, **kw):
            err = Exception("transport blew up")
            err.response = None          # as botocore sets it on some faults
            raise err

    client._client = Flaky()
    with pytest.raises(Exception, match="transport blew up"):
        client.complete_constrained(
            model_id="m", system="s", user="u", tool_name="submit",
            tool_schema={"type": "object", "properties": {}},
            description="d", temperature=0.0, max_tokens=64)
