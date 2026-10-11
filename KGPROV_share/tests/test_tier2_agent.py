"""Tier 2: the unbounded agent and live existence verification.

Two properties:

  * escalation issues model calls, and the loop is the model's to drive;
  * a well-formed identifier that does not exist is rejected, which
    needs an existence check against the live registry and not only
    against the local ontology.

No network, no AWS: the registry is faked.
"""

from __future__ import annotations

import pytest

from kgprov.llm import BedrockClient
from pipeline import phases, registry
from pipeline.loaders import MeshOntology, OntologyEntry
from pipeline.registry import EscalationCache, is_wellformed

from _fakes import FakeBedrock, text as _text, tool_use as _tool_use


# ---------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------

@pytest.fixture()
def onto():
    """Local ontology containing exactly one descriptor."""
    return MeshOntology(
        version="test",
        entries={"D000860": OntologyEntry(
            descriptor_ui="D000860", preferred_name="Hypoxia",
            synonyms=["Hypoxemia", "Anoxaemia"],
            tree_numbers=["C23.550.549"],
            entity_types={"Disease"})},
        surface_forms=[("hypoxia", "D000860")])


@pytest.fixture()
def no_live(monkeypatch):
    """Registry that knows nothing: every live lookup misses."""
    monkeypatch.setattr(registry, "resolve_live", lambda ui, **k: None)
    monkeypatch.setattr(phases, "resolve_live", lambda ui, **k: None)


@pytest.fixture()
def live_knows_D008687(monkeypatch):
    def fake(ui, **k):
        if ui == "D008687":
            return registry.LiveRecord(
                ui="D008687", preferred_name="Metformin",
                synonyms=["Glucophage", "Metformin Hydrochloride"],
                exists=True,
                source_url="https://id.nlm.nih.gov/mesh/D008687")
        return None
    monkeypatch.setattr(registry, "resolve_live", fake)
    monkeypatch.setattr(phases, "resolve_live", fake)


# ---------------------------------------------------------------------
#  Shape is not existence
# ---------------------------------------------------------------------

def test_a_fabricated_identifier_is_wellformed():
    """The premise of the whole verification step."""
    assert is_wellformed("D999999")
    assert is_wellformed("D008687")
    assert not is_wellformed("definitely-not-a-ui")


# ---------------------------------------------------------------------
#  Live verification
# ---------------------------------------------------------------------

def test_fabricated_identifier_is_rejected(onto, no_live):
    """A well-formed identifier known nowhere fails EXISTENCE."""
    ok, checks = phases._verify_tier2_live("D999999", onto, "hypoxemia",
                                           "Disease")
    assert ok is False
    assert checks["EXISTENCE"] is False


def test_malformed_identifier_fails_existence(onto, no_live):
    ok, checks = phases._verify_tier2_live("not-a-ui", onto, "x", "Disease")
    assert ok is False and checks["EXISTENCE"] is False


def test_locally_present_identifier_passes(onto, no_live):
    ok, checks = phases._verify_tier2_live("D000860", onto, "hypoxemia",
                                           "Disease")
    assert ok is True
    assert checks == {"EXISTENCE": True, "LEXICAL": True, "TYPE": True}


def test_existence_falls_through_to_the_live_registry(onto,
                                                      live_knows_D008687):
    """Absent locally but present in the live registry passes EXISTENCE."""
    ok, checks = phases._verify_tier2_live("D008687", onto, "metformin",
                                           "Chemical")
    assert checks["EXISTENCE"] is True
    assert ok is True


def test_existing_identifier_can_still_fail_lexical_agreement(
        onto, live_knows_D008687):
    """Exists, but denotes something else. A different failure from absence."""
    ok, checks = phases._verify_tier2_live("D008687", onto,
                                           "cerebral haemorrhage", "Chemical")
    assert checks["EXISTENCE"] is True
    assert checks["LEXICAL"] is False
    assert ok is False


def test_synonym_match_satisfies_lexical_agreement(onto, no_live):
    ok, checks = phases._verify_tier2_live("D000860", onto, "Anoxaemia",
                                           "Disease")
    assert checks["LEXICAL"] is True and ok is True


def test_type_disagreement_rejects(onto, no_live):
    ok, checks = phases._verify_tier2_live("D000860", onto, "hypoxemia",
                                           "Chemical")
    assert checks["TYPE"] is False and ok is False


def test_unknown_type_is_not_treated_as_incompatible(onto,
                                                     live_knows_D008687):
    """Live records carry no tree data. Unknown != incompatible."""
    _, checks = phases._verify_tier2_live("D008687", onto, "metformin",
                                          "Chemical")
    assert checks["TYPE"] is True


# ---------------------------------------------------------------------
#  Structural guard: the grounding path must not resolve by lookup
# ---------------------------------------------------------------------

def test_grounding_escalates_through_the_agent_not_the_lookup():
    """phase2_ground escalates through the agent, not `_tier2_lookup`.

    `_tier2_lookup` is a dict lookup against the prior MeSH release with
    no `llm` parameter. Nothing in the grounding path may call it.
    """
    import inspect
    src = inspect.getsource(phases.phase2_ground)
    assert "_run_tier2_agent(" in src, "escalation must go through the agent"
    assert "_tier2_lookup(" not in src, (
        "phase2_ground must not resolve an escalation by dictionary lookup")


def test_tier2_lookup_takes_no_model_and_so_cannot_be_agentic():
    """`_tier2_lookup` accepts no model, so it cannot act as an agent."""
    import inspect
    params = inspect.signature(phases._tier2_lookup).parameters
    assert "llm" not in params


def test_arm_decision_distinguishes_declining_from_not_being_asked():
    from pipeline.phases import (TIER1_DECLINED, TIER1_NOT_CALLED,
                                 TIER1_SELECTED)
    assert len({TIER1_SELECTED, TIER1_DECLINED, TIER1_NOT_CALLED}) == 3
    d = phases.Phase2ArmDecision(
        dedup_key=("x", "Disease"), arm="weak_arm", mapped_id=None,
        confidence="LOW", candidate_ids=[], escalated=True)
    # Absence of a decision is the safe default: a record that says
    # nothing must never be read as the model having said NO_MATCH.
    assert d.outcome == TIER1_NOT_CALLED


def test_arm_decision_carries_the_mentions_it_decided_for():
    """Without this the mention->decision join would have to be
    reconstructed by string matching on mention text."""
    d = phases.Phase2ArmDecision(
        dedup_key=("x", "Disease"), arm="weak_arm", mapped_id="D1",
        confidence="HIGH", candidate_ids=["D1"], escalated=False,
        covered_entity_ids=["E_a", "E_b", "E_c"])
    assert d.covered_entity_ids == ["E_a", "E_b", "E_c"]


# ---------------------------------------------------------------------
#  The agent
# ---------------------------------------------------------------------

def _client(tmp_path, responses):
    c = BedrockClient(region="us-east-1",
                      response_log_dir=str(tmp_path / "log"))
    c._client = FakeBedrock(responses)
    return c


def test_escalation_issues_model_calls(tmp_path, monkeypatch):
    """Escalation runs the agent loop: one tool call, then a submission."""
    monkeypatch.setattr(phases, "search_biomedical_web",
                        lambda q, **k: "Registry results:\n- D000860 Hypoxia")
    c = _client(tmp_path, [
        _tool_use("search_biomedical_web", {"query": "hypoxemia MeSH"}),
        _text("found D000860"),
        _tool_use("submit_tier2_result",
                  {"final_cui": "D000860",
                   "source_url": "https://id.nlm.nih.gov/mesh/D000860"},
                  stop="end_turn"),
    ])
    claim, url, trace = phases._run_tier2_agent(
        "weak_arm", "model", "hypoxemia", "Disease", llm=c)

    assert claim == "D000860"
    assert "D000860" in url
    assert trace["n_tool_calls"] == 1
    assert trace["termination_reason"] == "MODEL_CONCLUDED"


def test_the_model_composes_the_registry_query(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(phases, "search_biomedical_web",
                        lambda q, **k: seen.append(q) or "No registry ID found.")
    c = _client(tmp_path, [
        _tool_use("search_biomedical_web", {"query": "acute liver insult"}),
        _text("nothing"),
        _tool_use("submit_tier2_result", {"final_cui": "UNMAPPED"},
                  stop="end_turn"),
    ])
    phases._run_tier2_agent("weak_arm", "model", "acute liver insult",
                            "Disease", llm=c)
    assert seen == ["acute liver insult"]


def test_unmapped_submission_yields_no_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(phases, "search_biomedical_web",
                        lambda q, **k: "No registry ID found.")
    c = _client(tmp_path, [
        _text("no search needed"),
        _tool_use("submit_tier2_result", {"final_cui": "UNMAPPED"},
                  stop="end_turn"),
    ])
    claim, url, trace = phases._run_tier2_agent(
        "weak_arm", "model", "gibberish", "Disease", llm=c)
    assert claim is None
    assert trace["n_tool_calls"] == 0


def test_budget_exhaustion_is_distinguished_from_conclusion(tmp_path,
                                                           monkeypatch):
    """Two outcomes a lineage graph renders identically, different remedies."""
    monkeypatch.setattr(phases, "search_biomedical_web",
                        lambda q, **k: "No registry ID found.")
    responses = [_tool_use("search_biomedical_web", {"query": f"q{i}"})
                 for i in range(8)]
    responses.append(_tool_use("submit_tier2_result",
                               {"final_cui": "UNMAPPED"}, stop="end_turn"))
    c = _client(tmp_path, responses)
    _, _, trace = phases._run_tier2_agent(
        "weak_arm", "model", "x", "Disease", llm=c, max_tool_calls=2)
    assert trace["n_tool_calls"] == 2
    assert trace["termination_reason"] == "BUDGET_EXHAUSTED"


def test_cache_short_circuits_but_a_miss_runs_the_agent(tmp_path,
                                                        monkeypatch):
    monkeypatch.setattr(phases, "search_biomedical_web",
                        lambda q, **k: "Registry results:\n- D000860 Hypoxia")
    cache = EscalationCache(tmp_path / "esc.json")
    resp = [_tool_use("search_biomedical_web", {"query": "hypoxemia"}),
            _text("ok"),
            _tool_use("submit_tier2_result", {"final_cui": "D000860"},
                      stop="end_turn")]
    c = _client(tmp_path, list(resp))
    a = phases._run_tier2_agent("weak_arm", "model", "hypoxemia", "Disease",
                                llm=c, cache=cache)
    turns_used = len(c._client.requests)

    # second call: cache hit, no further provider traffic
    b = phases._run_tier2_agent("weak_arm", "model", "hypoxemia", "Disease",
                                llm=c, cache=cache)
    assert a[0] == b[0] == "D000860"
    assert b[2]["from_cache"] is True
    assert len(c._client.requests) == turns_used

    # different arm is a different key -> must run, not reuse
    c2 = _client(tmp_path, list(resp))
    phases._run_tier2_agent("strong_arm", "other-model", "hypoxemia",
                            "Disease", llm=c2, cache=cache)
    assert len(c2._client.requests) > 0, (
        "a different arm is a different escalation-cache key and must "
        "run the agent, not reuse the other arm's answer")
