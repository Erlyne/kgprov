"""Per-key sampling of the expensive arms.

Two properties are tested:

  * the admitted subset does not depend on the order keys are seen in,
    which is why the sampler hashes (seed, key) instead of drawing from
    an RNG stream: a run resumed under the same --run-id samples the same
    keys and replays them from the response log;
  * an arm skipped for a key produces no decision row at all, rather
    than a row carrying a fourth outcome value. `NOT_CALLED` means the
    candidate pool was empty, and "outside the sample" is a different
    fact with a different remedy.

No AWS: Tier 1, Tier 2 and retrieval are faked.
"""

from __future__ import annotations

import pytest

from pipeline import phases
from pipeline.phases import Mention, phase2_ground
from pipeline.retrieval import RetrievalResult
from pipeline.sampling import ArmSampler, rate_for_target


# ---------------------------------------------------------------------
#  The sampler in isolation
# ---------------------------------------------------------------------

def _keys(n: int) -> list[tuple[str, str]]:
    return [(f"term_{i}", "Chemical") for i in range(n)]


def test_a_census_arm_is_always_admitted():
    s = ArmSampler(arms={"strong_arm"}, rate=0.0)
    assert s.samples("strong_arm") and not s.samples("weak_arm")
    assert all(s.admits("weak_arm", k) for k in _keys(50))
    assert not any(s.admits("strong_arm", k) for k in _keys(50))


def test_the_same_key_gets_the_same_answer_every_time():
    """Keys recur across documents; the arm must not flip on the second
    sighting, or the pair of cells stops sharing a population."""
    s = ArmSampler(arms={"strong_arm"}, rate=0.5, seed=7)
    first = [s.admits("strong_arm", k) for k in _keys(200)]
    again = [s.admits("strong_arm", k) for k in _keys(200)]
    assert first == again


def test_admission_is_independent_of_order():
    """A resumed run admits the same keys whatever order it sees them in."""
    ks = _keys(300)
    fwd = ArmSampler(arms={"strong_arm"}, rate=0.3, seed=11)
    rev = ArmSampler(arms={"strong_arm"}, rate=0.3, seed=11)
    for k in ks:
        fwd.admits("strong_arm", k)
    for k in reversed(ks):
        rev.admits("strong_arm", k)
    assert fwd.admitted("strong_arm") == rev.admitted("strong_arm")


def test_a_fresh_sampler_reproduces_the_subset_from_the_seed_alone():
    ks = _keys(300)
    a = ArmSampler(arms={"strong_arm"}, rate=0.25, seed=3)
    b = ArmSampler(arms={"strong_arm"}, rate=0.25, seed=3)
    for k in ks:
        a.admits("strong_arm", k)
        b.admits("strong_arm", k)
    assert a.admitted("strong_arm") == b.admitted("strong_arm")


def test_a_different_seed_gives_a_different_subset():
    ks = _keys(300)
    a = ArmSampler(arms={"strong_arm"}, rate=0.25, seed=3)
    b = ArmSampler(arms={"strong_arm"}, rate=0.25, seed=4)
    for k in ks:
        a.admits("strong_arm", k)
        b.admits("strong_arm", k)
    assert a.admitted("strong_arm") != b.admitted("strong_arm")


def test_the_realised_fraction_tracks_the_rate():
    ks = _keys(4000)
    s = ArmSampler(arms={"strong_arm"}, rate=0.2, seed=1)
    n = sum(1 for k in ks if s.admits("strong_arm", k))
    assert 0.18 < n / len(ks) < 0.22


def test_rate_one_is_a_census():
    s = ArmSampler(arms={"strong_arm"}, rate=1.0)
    assert all(s.admits("strong_arm", k) for k in _keys(100))


def test_the_cap_is_a_hard_ceiling_and_says_when_it_bound():
    s = ArmSampler(arms={"strong_arm"}, rate=1.0, cap=10)
    admitted = sum(1 for k in _keys(500) if s.admits("strong_arm", k))
    assert admitted == 10
    assert s.summary()["cap_binding"]["strong_arm"] is True


def test_a_non_binding_cap_is_reported_as_such():
    """If the rate did the selecting, the sample is order-independent and
    the manifest should not imply otherwise."""
    s = ArmSampler(arms={"strong_arm"}, rate=0.1, cap=10_000)
    for k in _keys(200):
        s.admits("strong_arm", k)
    assert s.summary()["cap_binding"]["strong_arm"] is False


def test_summary_carries_what_the_manifest_needs():
    s = ArmSampler(arms={"strong_arm"}, rate=0.25, seed=42, cap=100)
    for k in _keys(80):
        s.admits("strong_arm", k)
    got = s.summary()
    assert got["sampled_arms"] == ["strong_arm"]
    assert got["seed"] == 42 and got["cap"] == 100
    assert got["rate"] == 0.25
    assert (got["keys_admitted"]["strong_arm"]
            + got["keys_rejected"]["strong_arm"]) == 80


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_an_impossible_rate_is_refused(rate):
    with pytest.raises(ValueError):
        ArmSampler(arms={"strong_arm"}, rate=rate)


def test_rate_for_target_sizes_the_sample_to_the_corpus():
    assert rate_for_target(400, 500, 4.6) == pytest.approx(400 / 2300)
    # A target larger than the corpus can supply is a census, not a rate
    # above one.
    assert rate_for_target(10_000, 5, 4.6) == 1.0


# ---------------------------------------------------------------------
#  phase2_ground under a sampler
# ---------------------------------------------------------------------

@pytest.fixture()
def stub_phase2(monkeypatch):
    """Retrieval, Tier 1 and Tier 2 replaced by call recorders."""
    calls = {"tier1": [], "tier2": []}

    def fake_retrieve(index, query, cat, params, recorder=None,
                      discriminators=None, compact_narrowing=True,
                      **kwargs):
        return RetrievalResult(
            query=query, category=cat, dense_uis=["D000001"],
            sparse_uis=["D000001"], fused_uis=["D000001"],
            typed_uis=["D000001"], reranked=[("D000001", 9.0)],
            final=[("D000001", 9.0)], suppressed_by_tau=False)

    def fake_pool(index, result, max_synonyms=4):
        return [{"id": ui, "preferred_name": ui, "synonyms": [],
                 "entity_types": ["Chemical"]} for ui, _ in result.final]

    def fake_tier1(arm, model_id, query, cat, candidates, **kw):
        calls["tier1"].append((arm, query))
        # LOW confidence, so every arm meets the escalation trigger and
        # the Tier 2 accounting is exercised on every key.
        return candidates[0]["id"], "LOW"

    def fake_tier2(arm, model_id, query, cat, **kw):
        calls["tier2"].append((arm, query))
        return None, "", {"n_tool_calls": 1,
                          "termination_reason": "MODEL_CONCLUDED"}

    monkeypatch.setattr(phases, "retrieve", fake_retrieve)
    monkeypatch.setattr(phases, "_candidate_pool", fake_pool)
    monkeypatch.setattr(phases, "_run_tier1", fake_tier1)
    monkeypatch.setattr(phases, "_run_tier2_agent", fake_tier2)
    return calls


def _mentions(n: int) -> list[Mention]:
    return [Mention(
        entity_id=f"E_{i}", doc_id="d1", chunk_id=0,
        start_char=0, end_char=1, doc_start_char=0,
        raw_span=f"term {i}", resolved_string=f"term {i}",
        resolution_type="NONE", semantic_category="Chemical")
        for i in range(n)]


def _ground(mentions, *, tier1_arms=(("weak_arm", "w"), ("strong_arm", "s")),
            **kw):
    return phase2_ground(
        mentions, index=None, params=None, ontology=None,
        ontology_prior=None, llm=None, tier1_arms=list(tier1_arms),
        primary_arm="weak_arm", recorder=None, **kw)


def test_without_a_sampler_every_arm_is_a_census(stub_phase2):
    _, decs = _ground(_mentions(20))
    per_arm = {a: sum(1 for d in decs if d.arm == a)
               for a in ("weak_arm", "strong_arm")}
    assert per_arm == {"weak_arm": 20, "strong_arm": 20}


def test_an_unsampled_key_produces_no_row_for_that_arm(stub_phase2):
    """Not a fourth outcome value: NOT_CALLED means the pool was
    empty, which is a different fact from an unsampled key."""
    s = ArmSampler(arms={"strong_arm"}, rate=0.3, seed=5)
    _, decs = _ground(_mentions(60), sampler=s)
    weak = [d for d in decs if d.arm == "weak_arm"]
    strong = [d for d in decs if d.arm == "strong_arm"]
    assert len(weak) == 60, "the primary arm stays a census"
    assert 0 < len(strong) < 60
    assert {d.dedup_key for d in strong} == s.admitted("strong_arm")
    assert all(d.outcome in {"SELECTED", "DECLINED", "NOT_CALLED"}
               for d in decs)


def test_skipping_an_arms_tier1_skips_its_tier2(stub_phase2):
    """Both strong cells land on one key set, which is what makes the
    strong row of the 2x2 a single population."""
    s = ArmSampler(arms={"strong_arm"}, rate=0.3, seed=5)
    _, decs = _ground(_mentions(60), sampler=s)
    t2_strong = {q for arm, q in stub_phase2["tier2"] if arm == "strong_arm"}
    t1_strong = {q for arm, q in stub_phase2["tier1"] if arm == "strong_arm"}
    assert t2_strong == t1_strong
    assert len(stub_phase2["tier2"]) == 60 + len(t2_strong)


def test_sampling_the_primary_arm_is_refused(stub_phase2):
    """Its decision stamps mention.cui; sampling it would leave mentions
    ungrounded rather than ungrounded-by-measurement."""
    s = ArmSampler(arms={"weak_arm"}, rate=0.3)
    with pytest.raises(ValueError, match="primary_arm"):
        _ground(_mentions(5), sampler=s)


def test_tier2_arms_can_be_a_strict_subset(stub_phase2):
    _, decs = _ground(_mentions(10), tier2_arms=["weak_arm"])
    assert {arm for arm, _ in stub_phase2["tier2"]} == {"weak_arm"}
    strong = [d for d in decs if d.arm == "strong_arm"]
    assert len(strong) == 10
    # Tier 1 met the trigger, but the unbounded cell did not run. Both
    # facts are recorded; collapsing them would make a budget decision
    # look like a null result.
    assert all(d.escalation_triggered and not d.escalated for d in strong)


def test_tier2_arms_outside_tier1_arms_is_refused(stub_phase2):
    with pytest.raises(ValueError, match="tier2_arms"):
        _ground(_mentions(5), tier1_arms=[("weak_arm", "w")],
                tier2_arms=["strong_arm"])


def test_an_empty_tier2_arms_list_runs_no_unbounded_cell(stub_phase2):
    _, decs = _ground(_mentions(10), tier2_arms=[])
    assert stub_phase2["tier2"] == []
    assert all(not d.escalated and d.escalation_triggered for d in decs)
