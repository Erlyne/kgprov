"""What the capture sidecar emits, checked on a miniature instrumented run.

Properties tested:
  * profile nodes are also typed with their core PROV class, so a
    consumer ignoring the kgp: namespace still reads a valid graph;
  * constraint-space membership is written twice, once plainly and once
    qualified with rank, score and selection;
  * attestation is not modelled as derivation;
  * narrowing records a stage label, cardinalities and a per-stage gold
    witness, including for compacted spaces;
  * IRIs are valid Turtle without post-processing.
"""

from __future__ import annotations

import pytest
from rdflib import RDF, Literal
from rdflib.namespace import XSD

from kgprov import (KGP, PROV, ENFORCE_DECODE_TIME, Recorder, iri, space)


@pytest.fixture()
def rec():
    """A miniature Phase-2 grounding, fully instrumented, no model."""
    r = Recorder("test001", model_id="test", ontology_version="mesh2025")
    r.agent("agent_evaluator", cls=KGP.PrimaryEvaluatorAgent)
    r.agent("agent_escalation", cls=KGP.EscalationSearchAgent,
            on_behalf_of="agent_evaluator")

    s0 = space("bm25_faiss_fused", [f"D{i:03d}" for i in range(100)],
               discriminators=("m1",))
    with r.narrowing("TYPE_FILTER", input_space=s0,
                     discriminators=("m1",)) as n:
        kept = [f"D{i:03d}" for i in range(40)]
        n.excluded([f"D{i:03d}" for i in range(40, 100)],
                   reason="TYPE_INCOMPATIBLE")
        s1 = n.output_space(space("type_filtered", kept,
                                  discriminators=("m1",)))

    with r.narrowing("CROSS_ENCODER_RERANK", input_space=s1,
                     discriminators=("m1",)) as n:
        top5 = kept[:5]
        n.excluded(kept[5:], reason="BELOW_TOP_K")
        s2 = n.output_space(space(
            "top5", top5, discriminators=("m1",),
            scores={v: 0.9 - i * 0.1 for i, v in enumerate(top5)},
            selected="D000"))

    with r.generation("phase2_tier1", agent="agent_evaluator",
                      plan="plan_tier1", prompt_hash="abc123",
                      autonomy_level=3, discriminators=("m1",)) as call:
        call.constrained_by(s2, enforcement=ENFORCE_DECODE_TIME,
                            enforcement_artifact="sha1:fake-schema")
        call.generated("cui_D000", cls=KGP.CanonicalConceptID,
                       mappingTier="LOCAL", mappingConfidence="HIGH")

    with r.attestation(agent="agent_escalation", subject="cui_D000",
                       discriminators=("m1",)) as att:
        att.against("https://registry.example/D000",
                    specialization_of="registry_general")
        att.outcome(KGP.Confirmed, repeatable=False)

    r._spaces_for_test = {"s0": s0, "s1": s1, "s2": s2}
    return r


# ---------------------------------------------------------------------
#  Space semantics
# ---------------------------------------------------------------------

def test_narrowing_actually_narrows(rec):
    s = rec._spaces_for_test
    assert len(s["s0"].values) == 100
    assert len(s["s1"].values) == 40
    assert len(s["s2"].values) == 5


def test_space_membership_is_accurate(rec):
    s2 = rec._spaces_for_test["s2"]
    assert s2.contains("D000")
    assert not s2.contains("D050")


def test_space_id_depends_on_derivation_not_contents(rec):
    """Two spaces with the same members but different provenance differ."""
    a = space("top5", ["D1"], discriminators=("m1",))
    b = space("top5", ["D1"], discriminators=("m2",))
    assert a.space_id != b.space_id


# ---------------------------------------------------------------------
#  Narrowing stages are labels, not subclasses
# ---------------------------------------------------------------------

def test_narrowing_records_a_stage_label(rec):
    stages = {str(o) for o in rec.g.objects(None, KGP.narrowingStage)}
    assert stages == {"TYPE_FILTER", "CROSS_ENCODER_RERANK"}


def test_narrowing_subclasses_are_gone(rec):
    """No narrowing activity is typed with a per-stage subclass.

    The stage is recorded with kgp:narrowingStage: "at which stage was the
    answer lost" is answered by the stage identity plus the cardinalities.
    """
    for gone in ("DenseRetrieval", "SparseRetrieval", "RankFusion",
                 "TypeFilter", "CrossEncoderRerank", "ScoreFloorFilter"):
        assert not list(rec.g.subjects(RDF.type, KGP[gone])), (
            f"kgp:{gone} should no longer be instantiated")


def test_every_narrowing_is_still_a_prov_activity(rec):
    acts = list(rec.g.subjects(RDF.type, KGP.NarrowingActivity))
    assert len(acts) == 2
    for a in acts:
        assert (a, RDF.type, PROV.Activity) in rec.g
        assert list(rec.g.objects(a, KGP.inputCardinality))
        assert list(rec.g.objects(a, KGP.outputCardinality))


# ---------------------------------------------------------------------
#  Gold witness: stage attribution as a graph query
# ---------------------------------------------------------------------

def _chain(gold):
    """Two-stage chain where gold survives the first and dies at the second."""
    r = Recorder("t", model_id="m", ontology_version="v")
    s0 = space("fused", [f"D{i:03d}" for i in range(10)], discriminators=("m",))
    with r.narrowing("TYPE_FILTER", input_space=s0, gold=gold,
                     discriminators=("m",)) as n:
        s1 = n.output_space(space("typed", [f"D{i:03d}" for i in range(6)],
                                  discriminators=("m",)))
    with r.narrowing("CROSS_ENCODER_RERANK", input_space=s1, gold=gold,
                     discriminators=("m",)) as n:
        n.output_space(space("top3", [f"D{i:03d}" for i in range(3)],
                            discriminators=("m",)))
    return r


def _survival(r):
    return {str(next(r.g.objects(a, KGP.narrowingStage))):
            bool(next(r.g.objects(a, KGP.goldSurvived)))
            for a in r.g.subjects(RDF.type, KGP.NarrowingActivity)}


def test_gold_survival_is_recorded_per_stage():
    """Gold survival is recorded per stage, so attribution is a graph query."""
    assert _survival(_chain("D004")) == {"TYPE_FILTER": True,
                                         "CROSS_ENCODER_RERANK": False}


def test_gold_that_survives_everything_is_recorded_as_such():
    assert all(_survival(_chain("D001")).values())


def test_gold_lost_at_the_first_stage_is_attributed_there():
    assert _survival(_chain("D007")) == {"TYPE_FILTER": False,
                                         "CROSS_ENCODER_RERANK": False}


def test_no_gold_witness_when_no_ground_truth(rec):
    """Production traces have no gold; the term must stay absent, not false."""
    assert not list(rec.g.objects(None, KGP.goldSurvived))


# ---------------------------------------------------------------------
#  Graceful degradation for plain PROV consumers
# ---------------------------------------------------------------------

def test_membership_is_written_both_plainly_and_qualified(rec):
    """A vanilla PROV consumer must still see a collection with members."""
    plain = list(rec.g.subjects(PROV.hadMember, None))
    qualified = list(rec.g.subjects(KGP.qualifiedMembership, None))
    assert plain, "no prov:hadMember - vanilla consumers see nothing"
    assert qualified, "no kgp:qualifiedMembership - rank/score lost"


def test_qualified_membership_carries_rank_and_selection(rec):
    found_selected = False
    for qm in rec.g.objects(None, KGP.qualifiedMembership):
        if (qm, KGP.wasSelected, Literal(True)) in rec.g:
            found_selected = True
    assert found_selected, "no member marked selected"


def test_constraint_space_is_a_prov_collection(rec):
    for s in rec.g.subjects(RDF.type, KGP.ConstraintSpace):
        assert (s, RDF.type, PROV.Collection) in rec.g, (
            "ConstraintSpace must specialise prov:Collection so a vanilla "
            "consumer reads it as a collection")


def test_derived_and_static_spaces_are_distinguished(rec):
    derived = set(rec.g.subjects(RDF.type, KGP.DerivedConstraintSpace))
    assert derived, "the static/derived split is the thesis's core claim"


# ---------------------------------------------------------------------
#  Attestation is not derivation
# ---------------------------------------------------------------------

def test_attestation_does_not_assert_derivation(rec):
    """An identifier checked against a registry was not derived from it."""
    for att in rec.g.subjects(RDF.type, KGP.Attestation):
        for src in rec.g.objects(att, KGP.attestedAgainst):
            assert (None, PROV.wasDerivedFrom, src) not in rec.g, (
                "attestation must not imply lineage from the authority")


def test_attestation_records_irreplayability(rec):
    outcomes = list(rec.g.subjects(KGP.isRepeatable, Literal(False)))
    assert outcomes, "consuming mutable external state must be recorded"


def test_attested_source_is_time_indexed(rec):
    """'The web' is not an entity whose membership can be asserted."""
    for att in rec.g.subjects(RDF.type, KGP.Attestation):
        for src in rec.g.objects(att, KGP.attestedAgainst):
            assert list(rec.g.objects(src, PROV.specializationOf)), (
                "attested source must specialise a general resource")


# ---------------------------------------------------------------------
#  Serialisation
# ---------------------------------------------------------------------

def test_serializes_to_parseable_turtle(rec, tmp_path):
    out = tmp_path / "prov.ttl"
    rec.serialize(str(out))
    assert out.exists() and out.stat().st_size > 0

    from rdflib import Graph
    reparsed = Graph()
    reparsed.parse(str(out), format="turtle")
    assert len(reparsed) == len(rec.g)


def test_graph_is_not_trivially_small(rec):
    assert len(rec) > 200


# ---------------------------------------------------------------------
#  Compaction must not corrupt the gold witness
# ---------------------------------------------------------------------

def test_compacted_space_witnesses_gold_against_true_membership():
    """The gold witness is taken from true_members, not the stand-in.

    Compaction records empty-member spaces to keep the graph small. A
    witness computed from the compacted stand-in would report gold as lost
    at every stage.
    """
    r = Recorder("t", model_id="m", ontology_version="v")
    s0 = space("fused", [f"D{i:03d}" for i in range(10)], discriminators=("m",))
    with r.narrowing("TYPE_FILTER", input_space=s0, gold="D002",
                     discriminators=("m",)) as n:
        # compacted: no members recorded, but D002 did survive
        n.output_space(space("typed", [], discriminators=("m",)),
                       true_members=[f"D{i:03d}" for i in range(6)])
    survived = [bool(o) for o in r.g.objects(None, KGP.goldSurvived)]
    assert survived == [True], "witness taken against the compacted stand-in"


def test_compacted_space_still_reports_a_genuine_loss():
    r = Recorder("t", model_id="m", ontology_version="v")
    s0 = space("fused", [f"D{i:03d}" for i in range(10)], discriminators=("m",))
    with r.narrowing("SCORE_FLOOR", input_space=s0, gold="D009",
                     discriminators=("m",)) as n:
        n.output_space(space("final", [], discriminators=("m",)),
                       true_members=[])
    assert [bool(o) for o in r.g.objects(None, KGP.goldSurvived)] == [False]


def test_compacted_space_reports_its_true_size():
    """A compacted space records the size of true_members, not 0.

    Compaction writes empty-member spaces to keep the graph small, so
    sizing a space by its member list would record `kgp:cardinality 0`
    for every compacted retrieval space.
    """
    from rdflib.namespace import XSD

    r = Recorder(run_id="t_card")
    big = space("index", [f"D{i:04d}" for i in range(200)],
                discriminators=("q",))
    with r.narrowing("DENSE_RETRIEVAL", input_space=big,
                     discriminators=("q",)) as n:
        # The compacted stand-in: no members, real membership passed aside.
        compact = space("dense_top_k", [], discriminators=("dense", "q"))
        n.output_space(compact, true_members=[f"D{i:04d}" for i in range(37)])

    sizes = [int(o) for o in r.g.objects(iri(compact.space_id),
                                         KGP.cardinality)]
    assert sizes == [37], f"compacted space reported {sizes}, expected [37]"


def test_true_members_is_not_consumed_before_the_gold_witness():
    """A generator passed as true_members still reaches the gold witness.

    `true_members` is read twice: once to size the space, once to test
    whether gold survived. Consuming it on the first read would make the
    witness false at every stage.
    """
    from rdflib import Literal
    from rdflib.namespace import XSD

    r = Recorder(run_id="t_gen")
    big = space("index", ["D1", "D2", "D3"], discriminators=("q",))
    with r.narrowing("DENSE_RETRIEVAL", input_space=big, gold="D2",
                     discriminators=("q",)) as n:
        compact = space("dense_top_k", [], discriminators=("dense", "q"))
        n.output_space(compact, true_members=(x for x in ["D1", "D2"]))
        act = n.uri

    survived = list(r.g.objects(act, KGP.goldSurvived))
    assert survived == [Literal(True, datatype=XSD.boolean)], survived
    sizes = [int(o) for o in r.g.objects(iri(compact.space_id),
                                         KGP.cardinality)]
    assert sizes == [2], sizes


# ---------------------------------------------------------------------
#  IRIs must be valid Turtle without a post-process
# ---------------------------------------------------------------------

@pytest.mark.parametrize("local", [
    "cui_weak_arm_D013134_[3h]spiperon_Chemical",
    "dedup_alpha{beta}_Chemical",
    'quote"inside',
    "pipe|char", "caret^char", "backtick`char",
    "angle<bracket>",
])
def test_iri_percent_encodes_turtle_forbidden_characters(local):
    """Characters Turtle forbids in IRIs are percent-encoded.

    rdflib serialises them without complaint, but a strict parser such as
    pyoxigraph rejects the whole file. Local names come from corpus text,
    and biomedical names often contain brackets, e.g. [3H]spiperone.
    """
    u = str(iri(local))
    assert not any(c in u for c in ' <>{}|^`"[]'), u
    assert u.startswith("https://w3id.org/kgprov/run/")


def test_a_graph_with_bracketed_names_round_trips_through_a_strict_parser():
    """End to end: write, serialize, reparse strictly."""
    pyoxigraph = pytest.importorskip("pyoxigraph")

    r = Recorder(run_id="t_iri")
    r.entity("cui_weak_arm_D013134_[3H]spiperone_Chemical",
             cls=KGP.CanonicalConceptID, cuiValue="D013134")
    ttl = r.g.serialize(format="turtle")

    store = pyoxigraph.Store()
    store.bulk_load(ttl.encode("utf-8"), pyoxigraph.RdfFormat.TURTLE)
    assert len(store) > 0


def test_ordinary_names_are_left_alone():
    """Encoding must not churn the identifiers that were already fine."""
    assert str(iri("cui_D008687")).endswith("/cui_D008687")


def test_a_generated_space_is_sized_like_the_activity_that_made_it(rec):
    """Each narrowing records the same size on the activity and its space.

    `kgp:outputCardinality` on the activity and `kgp:cardinality` on the
    space it generated describe the same set, so they must agree.
    """
    for act in set(rec.g.subjects(RDF.type, KGP.NarrowingActivity)):
        out_card = [int(o) for o in rec.g.objects(act, KGP.outputCardinality)]
        spaces = [s for s in rec.g.subjects(PROV.wasGeneratedBy, act)]
        assert len(out_card) == 1, (act, out_card)
        for sp in spaces:
            sizes = [int(o) for o in rec.g.objects(sp, KGP.cardinality)]
            stage = next(rec.g.objects(act, KGP.narrowingStage))
            assert sizes == out_card, (
                f"{stage}: space says {sizes}, activity says {out_card}")


def test_compact_rerank_records_the_cut_not_the_pool_it_scored():
    """A hand-built compact CROSS_ENCODER_RERANK record, with
    `true_members` set to the kept top-k.

    With that argument the space and the activity agree on the size.
    `pipeline.retrieval` passes the whole reranked pool as `true_members`
    instead, so the space is sized by the pool and the activity by the
    cut. This test does not exercise that call; the xfail test below does.
    """
    from pipeline.retrieval import _fix_cardinality

    reranked = [(f"D{i:04d}", 1.0 - i * 0.01) for i in range(86)]
    top_k = 5

    r = Recorder(run_id="t_rerank")
    typed = space("typed", [], discriminators=("m1",))
    with r.narrowing("CROSS_ENCODER_RERANK", input_space=typed,
                     gold="D0007", discriminators=("m1",)) as n:
        kept = [u for u, _ in reranked[:top_k]]
        out = space("reranked", [], discriminators=("rerank", "m1"))
        n.output_space(out, true_members=kept)
        _fix_cardinality(r, n.uri, 86, len(kept))
        act = n.uri

    space_size = [int(o) for o in r.g.objects(iri(out.space_id),
                                              KGP.cardinality)]
    act_out = [int(o) for o in r.g.objects(act, KGP.outputCardinality)]
    assert space_size == [top_k], space_size
    assert act_out == [top_k], act_out

    # gold at rank 8 was scored but cut, so the witness must say it did
    # not survive this stage -- that is the row CQ3 attributes.
    survived = [bool(o) for o in r.g.objects(act, KGP.goldSurvived)]
    assert survived == [False], survived


class _StubReranker:
    """Scores by descending index position, so gold's rank is controllable."""

    def predict(self, pairs):
        return [1.0 - i * 0.01 for i in range(len(pairs))]


class _StubIndex:
    """Just enough MeshIndex surface for `retrieve` to run offline."""

    version = "mesh_stub"

    def __init__(self, n=120):
        self.surface_forms = [(f"name{i}", f"D{i:04d}") for i in range(n)]
        self.types_by_ui = {f"D{i:04d}": {"Chemical"} for i in range(n)}
        self._reranker = _StubReranker()

    def preferred_name(self, ui):
        return f"name{ui}"

    def dense_search(self, query, k):
        return [(f"D{i:04d}", 1.0 - i * 0.001) for i in range(min(k, 120))]

    def sparse_search(self, query, k):
        return [(f"D{i:04d}", 1.0 - i * 0.001)
                for i in range(min(k, 120) - 1, -1, -1)]


@pytest.mark.xfail(
    strict=False,
    reason="Known capture defect: in compact mode the reranker's output "
           "space is stamped with the reranker's input cardinality "
           "(pipeline/retrieval.py). Kept so this code reproduces the "
           "released graph.")
def test_the_real_retrieval_chain_agrees_with_itself_about_every_size():
    """Drive `retrieve` and check every size in the chain end to end.

    Each stage writes the same set twice (`outputCardinality` on the
    activity, `cardinality` on the space it generated) and the next stage
    reads it a third time as its own `inputCardinality`. All three must
    agree. Expected to fail at CROSS_ENCODER_RERANK; see the xfail reason.
    """
    from pipeline.retrieval import RetrievalParams, retrieve

    r = Recorder(run_id="t_chain")
    r.agent("agent_retrieval", cls=KGP.PrimaryEvaluatorAgent)
    retrieve(_StubIndex(), "name0007", "Chemical",
             RetrievalParams(score_floor=-100.0),
             recorder=r, discriminators=("m1",), gold="D0007")

    seen = 0
    for act in set(r.g.subjects(RDF.type, KGP.NarrowingActivity)):
        stage = str(next(r.g.objects(act, KGP.narrowingStage)))
        out_card = [int(o) for o in r.g.objects(act, KGP.outputCardinality)]
        for sp in r.g.subjects(PROV.wasGeneratedBy, act):
            sizes = [int(o) for o in r.g.objects(sp, KGP.cardinality)]
            assert sizes == out_card, (
                f"{stage}: space says {sizes}, activity says {out_card}")
            seen += 1
            # and the stage that consumed this space must have read the
            # same number as its input
            for nxt in r.g.subjects(PROV.used, sp):
                if (nxt, RDF.type, KGP.NarrowingActivity) not in r.g:
                    continue
                in_card = [int(o) for o in
                           r.g.objects(nxt, KGP.inputCardinality)]
                nstage = str(next(r.g.objects(nxt, KGP.narrowingStage)))
                assert in_card == sizes, (
                    f"{nstage} read {in_card} from a space of {sizes} "
                    f"produced by {stage}")
    assert seen >= 5, f"only {seen} generated spaces checked"
