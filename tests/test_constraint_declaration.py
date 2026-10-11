"""Every constraint space must be declared to the decoder.

Each phase schema must carry its admissible set as an `enum`, and every
phase must call the constrained path. With a shape-only schema
({"type": "array"}, {"type": "string"}) the recorded ConstraintSpace
entities would describe a constraint that was never applied at decode
time.
"""

from __future__ import annotations

import inspect
import re

import pytest

from pipeline import phases


def _items(schema, key):
    return schema["properties"][key]["items"]["properties"]


# ---------------------------------------------------------------------
#  Every phase declares its admissible set
# ---------------------------------------------------------------------

def test_phase1_declares_the_static_category_space():
    """The pipeline's only static space, and the contrast that gives
    'derived' its meaning everywhere else."""
    props = _items(phases.PHASE1_SCHEMA, "entities")
    assert props["semantic_category"]["enum"] == ["CHEMICAL", "DISEASE"]
    assert props["resolution_type"]["enum"]


def test_tier1_compiles_the_candidate_enum_per_call():
    """The derived space: manufactured at runtime, gone after the run."""
    s = phases._tier1_schema([{"id": "D000860"}, {"id": "D008687"}])
    enum = s["properties"]["mapped_id"]["enum"]
    assert enum == ["D000860", "D008687", "NO_MATCH"]


def test_tier1_enum_tracks_the_candidates_it_was_given():
    a = phases._tier1_schema([{"id": "D1"}])["properties"]["mapped_id"]["enum"]
    b = phases._tier1_schema([{"id": "D2"}])["properties"]["mapped_id"]["enum"]
    assert a != b, "a per-call enum that does not vary is not derived"


def test_tier1_always_offers_a_way_to_decline():
    """NO_MATCH is always in the enum, so the model can decline."""
    s = phases._tier1_schema([{"id": "D1"}])
    assert "NO_MATCH" in s["properties"]["mapped_id"]["enum"]


def test_phase3_enum_is_derived_from_phase2_output():
    """Derived spaces chain: an upstream error propagates as a corrupted
    admissible set, not merely as bad data."""
    props = _items(phases._phase3_schema(["D1", "D2"]),
                   "extracted_relationships")
    assert props["subject_cui"]["enum"] == ["D1", "D2"]
    assert props["object_cui"]["enum"] == ["D1", "D2"]


def test_phase3_polarity_cannot_be_unparseable():
    """Coercing an unrecognised polarity to POSITIVE would turn a negation
    into an assertion. With the enum declared, the value cannot be
    produced at all."""
    props = _items(phases._phase3_schema(["D1"]), "extracted_relationships")
    assert props["polarity"]["enum"] == ["POSITIVE", "NEGATIVE"]
    assert props["modality"]["enum"] == ["ASSERTED", "HYPOTHESIZED",
                                        "PROPOSED"]


def test_phase4_declares_omega_plus_the_escape_codes():
    props = _items(phases._phase4_schema(["causes", "treats"]),
                   "normalized_edges")
    enum = props["normalized_predicate"]["enum"]
    assert enum[:2] == ["causes", "treats"]
    assert "PRED:NO_INTERACTION" in enum and "PRED:GARBLED" in enum


def test_phase4_enum_tracks_the_bucket():
    a = phases._phase4_schema(["causes"])
    b = phases._phase4_schema(["is_treated_by"])
    ea = _items(a, "normalized_edges")["normalized_predicate"]["enum"]
    eb = _items(b, "normalized_edges")["normalized_predicate"]["enum"]
    assert ea != eb


# ---------------------------------------------------------------------
#  Nothing may bypass the constrained path
# ---------------------------------------------------------------------

def test_no_phase_issues_an_unconstrained_call():
    """complete() sends the schema nowhere: it prompts and checks after.

    Any phase still using it is generating without a declared admissible
    set, which makes every ConstraintSpace it records a description of a
    constraint that was not applied.
    """
    src = inspect.getsource(phases)
    unconstrained = re.findall(r"llm\.complete\(", src)
    assert not unconstrained, (
        f"{len(unconstrained)} phase call(s) bypass constrained decoding")
    assert len(re.findall(r"llm\.complete_constrained\(", src)) == 4


def test_the_unconstrained_schema_is_retained_only_for_comparison():
    """The shape-only schema declares no enum and no phase sends it as a
    constraint."""
    s = phases.PHASE2_SCHEMA_UNCONSTRAINED
    assert "enum" not in str(s), "the comparison arm must stay unconstrained"
    src = inspect.getsource(phases)
    assert "tool_schema=PHASE2_SCHEMA_UNCONSTRAINED" not in src


@pytest.mark.parametrize("builder,args", [
    (lambda: phases._tier1_schema([{"id": "D1"}]), None),
    (lambda: phases._phase3_schema(["D1"]), None),
    (lambda: phases._phase4_schema(["causes"]), None),
])
def test_every_compiled_schema_contains_an_enum(builder, args):
    assert "enum" in str(builder())


# ---------------------------------------------------------------------
#  The enforcement claim must be checkable against what was sent
# ---------------------------------------------------------------------

def test_enforcement_artifact_matches_the_schema_the_provider_received(tmp_path):
    """The recorded enforcement artifact is a schema the provider received.

    The hash stored on the space as `kgp:enforcementArtifact` must equal
    the hash of a tool schema actually sent, so a change that stops
    sending the enum, or sends a different one, fails this test.
    """
    import json

    from kgprov import KGP, Recorder, ids
    from pipeline.chunking import chunk_document
    from pipeline.loaders import Document
    from pipeline.phases import phase1_extract, register_agents
    from kgprov.llm import BedrockClient

    from _fakes import FakeBedrock, tool_use

    title = "Aspirin causes ulcers."
    abstract = "Aspirin induced gastric ulcers in rats."
    # Single-space separator between title and abstract, as in BC5CDR.
    doc = Document(pmid="1", title=title, abstract=abstract,
                   full_text=f"{title} {abstract}", mentions=[])
    chunks = chunk_document(doc.full_text, max_chunk_tokens=512,
                            sentence_overlap=1)
    span = "Aspirin"
    start = chunks[0].text.index(span)
    fake = FakeBedrock([
        tool_use("submit_entities", {"entities": [{
            "raw_span": span, "resolved_string": span,
            "semantic_category": "CHEMICAL", "resolution_type": "NONE",
            "start_char": start, "end_char": start + len(span)}]})
        for _ in chunks])
    llm = BedrockClient(region="us-east-1",
                        response_log_dir=str(tmp_path / "log"))
    llm._client = fake

    rec = Recorder(run_id="t_enforce")
    register_agents(rec)
    mentions, _ = phase1_extract(doc, chunks, llm=llm, model_id="m",
                                 recorder=rec, temperature=0.0,
                                 max_tokens=256)
    assert mentions, "fixture must produce a mention for the test to mean anything"

    sent = [r["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
            for r in fake.requests]
    assert sent, "no schema reached the provider"
    expected = {ids.content_hash(json.dumps(s, sort_keys=True)) for s in sent}

    recorded = {str(o) for o in rec.g.objects(None, KGP.enforcementArtifact)}
    assert recorded, "no enforcementArtifact was recorded"
    assert recorded <= expected, (
        f"recorded artifact {recorded} is not a schema that was sent "
        f"{expected} -- the enforcement claim does not describe the call")

    modes = {str(o) for o in rec.g.objects(None, KGP.constraintEnforcement)}
    assert modes == {"DECODE_TIME"}, modes


def test_enforcement_mode_must_be_one_of_the_three():
    """An unrecognised mode is a bug, not a new category."""
    from kgprov import Recorder, space

    rec = Recorder(run_id="t_enforce_bad")
    with rec.generation("p", agent="a", discriminators=("x",)) as call:
        with pytest.raises(ValueError, match="enforcement must be one of"):
            call.constrained_by(space("k", ["A"], discriminators=("x",)),
                                enforcement="PROBABLY_ENFORCED")
