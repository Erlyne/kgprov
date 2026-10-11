"""Content-addressed identifier properties.

Three properties the competency questions rely on: stability across
re-extraction, independence from iteration order, and convergence of
unmappable terms on one provisional node.
"""

from __future__ import annotations

from kgprov import ids


# ---------------------------------------------------------------------
#  norm()
# ---------------------------------------------------------------------

def test_norm_collapses_case_and_whitespace():
    assert ids.norm("TP53") == ids.norm("tp53") == ids.norm("  TP53  ")
    assert ids.norm("alpha   beta") == "alpha beta"


def test_norm_strips_terminal_punctuation_only():
    assert ids.norm("aspirin.") == "aspirin"
    assert ids.norm("aspirin;") == "aspirin"
    # Internal punctuation is meaningful in chemical names and must stay.
    assert ids.norm("1,2-dimethyl") == "1,2-dimethyl"


def test_norm_is_nfkc():
    # Fullwidth digits normalise to ASCII under NFKC.
    assert ids.norm("ＡＢ") == "ab"


def test_norm_is_idempotent():
    for s in ["TP53", " Aspirin. ", "1,2-dimethyl", "ＡＢ"]:
        assert ids.norm(ids.norm(s)) == ids.norm(s)


# ---------------------------------------------------------------------
#  Identifier stability
# ---------------------------------------------------------------------

def test_entity_id_is_stable_across_calls():
    a = ids.entity_id("123", 0, 5, 12, "aspirin")
    b = ids.entity_id("123", 0, 5, 12, "aspirin")
    assert a == b
    assert a.startswith("E_")


def test_entity_id_is_sensitive_to_every_component():
    base = ids.entity_id("123", 0, 5, 12, "aspirin")
    assert ids.entity_id("124", 0, 5, 12, "aspirin") != base
    assert ids.entity_id("123", 1, 5, 12, "aspirin") != base
    assert ids.entity_id("123", 0, 6, 12, "aspirin") != base
    assert ids.entity_id("123", 0, 5, 13, "aspirin") != base
    assert ids.entity_id("123", 0, 5, 12, "asprin") != base


def test_unmapped_id_converges_on_one_node():
    """The same ungrounded term yields one provisional node."""
    a = ids.unmapped_id("Foo Bar", "Chemical")
    b = ids.unmapped_id("  foo   bar  ", "Chemical")
    assert a == b, "normalisation must precede hashing"
    assert a.startswith("UNMAPPED:")


def test_unmapped_id_separates_categories():
    assert (ids.unmapped_id("cold", "Chemical")
            != ids.unmapped_id("cold", "Disease"))


def test_dedup_key_separates_categories():
    """Same string, different category: two dedup keys."""
    assert (ids.dedup_key("cold", "Chemical")
            != ids.dedup_key("cold", "Disease"))


def test_dedup_key_normalises_the_string():
    assert ids.dedup_key(" Cold ", "Disease") == ids.dedup_key("cold", "Disease")


def test_space_id_distinguishes_derivation_not_contents():
    """Two spaces with identical members but different provenance differ."""
    a = ids.space_id("top_k", "mention_1")
    b = ids.space_id("top_k", "mention_2")
    assert a != b
    assert ids.space_id("top_k", "mention_1") == a


def test_ids_are_not_positional_counters():
    """Identifiers must not depend on the order they were minted in."""
    forward = [ids.entity_id("d", 0, i, i + 3, f"m{i}") for i in range(5)]
    backward = [ids.entity_id("d", 0, i, i + 3, f"m{i}")
                for i in reversed(range(5))][::-1]
    assert forward == backward


def test_repeated_spans_collapse_to_one_mention():
    """The model may list the same occurrence several times; it is one mention.

    `entity_id` is a hash of (doc, chunk, offsets, span), so records at
    identical offsets are the same mention by construction. Keeping both
    would inflate the mention total, the join denominator and the
    decision-transfer share.
    """
    from pipeline.chunking import Chunk
    from pipeline.phases import _validate_phase1_entities

    text = "NPY modulates NPY release."
    chunk = Chunk(chunk_id=0, text=text, doc_offset=0)
    ent = {"raw_span": "NPY", "resolved_string": "NPY",
           "semantic_category": "CHEMICAL", "resolution_type": "NONE",
           "start_char": 0, "end_char": 3}
    rejected, dups = [], []
    kept = _validate_phase1_entities(text, [dict(ent) for _ in range(9)],
                                     "doc1", chunk, rejected,
                                     duplicates_out=dups)

    assert len(kept) == 1, f"9 identical spans became {len(kept)} mentions"
    assert len(dups) == 8, "the collapse must be counted, not silent"
    # A repeat is not a rejection: the span was valid and produced twice.
    # The eight reason codes stay eight.
    assert rejected == []


def test_distinct_offsets_are_distinct_mentions():
    """The collapse must key on identity, not on surface form."""
    from pipeline.chunking import Chunk
    from pipeline.phases import _validate_phase1_entities

    text = "NPY modulates NPY release."
    chunk = Chunk(chunk_id=0, text=text, doc_offset=0)
    base = {"raw_span": "NPY", "resolved_string": "NPY",
            "semantic_category": "CHEMICAL", "resolution_type": "NONE"}
    ents = [{**base, "start_char": 0, "end_char": 3},
            {**base, "start_char": 14, "end_char": 17}]
    rejected, dups = [], []
    kept = _validate_phase1_entities(text, ents, "doc1", chunk, rejected,
                                     duplicates_out=dups)
    assert len(kept) == 2 and dups == []


def test_provisional_flag_is_derived_not_remembered():
    """`is_provisional` is computed from the endpoint identifiers at
    construction: a relationship or edge with at least one UNMAPPED
    endpoint is provisional."""
    from pipeline.phases import KGEdge, Relationship

    grounded = Relationship(
        relation_id="R1", doc_id="d", chunk_id=0,
        subject_entity_id="E1", object_entity_id="E2",
        subject_cui="D001", object_cui="D002",
        raw_action_phrase="causes", action_start_char=0, action_end_char=6,
        polarity="POSITIVE", modality="ASSERTED", conditions={})
    assert grounded.is_provisional is False

    escalated = Relationship(
        relation_id="R2", doc_id="d", chunk_id=0,
        subject_entity_id="E1", object_entity_id="E2",
        subject_cui="D001", object_cui="UNMAPPED:56c9f82f8814",
        raw_action_phrase="causes", action_start_char=0, action_end_char=6,
        polarity="POSITIVE", modality="ASSERTED", conditions={})
    assert escalated.is_provisional is True, "one unmapped endpoint is enough"

    edge = KGEdge(subject_cui="UNMAPPED:abc", normalized_predicate="causes",
                  object_cui="D002", polarity="POSITIVE", modality="ASSERTED",
                  conditions={}, negated=False, evidence_count=1,
                  supporting_relation_ids=["R2"])
    assert edge.is_provisional is True
