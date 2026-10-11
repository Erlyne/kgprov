"""BC5CDR parsing invariants.

The offset convention is checked against the corpus because an error in
it would not raise; it would make every span check compare the wrong
characters.
"""

from __future__ import annotations

import pytest

from pipeline.loaders import parse_pubtator_file, _split_mesh_id


# ---------------------------------------------------------------------
#  Compound gold identifiers
# ---------------------------------------------------------------------

def test_split_mesh_id_takes_first_of_compound():
    assert _split_mesh_id("D000001|D000002") == ("D000001", True)


def test_split_mesh_id_flags_absent_gold():
    assert _split_mesh_id("-1") == (None, False)
    assert _split_mesh_id("") == (None, False)


def test_split_mesh_id_passes_through_simple():
    assert _split_mesh_id("D000001") == ("D000001", False)


def test_split_mesh_id_tolerates_surrounding_space():
    assert _split_mesh_id("  D000001  ") == ("D000001", False)


# ---------------------------------------------------------------------
#  The offset invariant
# ---------------------------------------------------------------------

def test_offsets_index_title_space_abstract(bc5cdr_dev):
    """full_text[start:end] must equal the annotated mention, exactly.

    BC5CDR offsets index `title + " " + abstract` with a single-space
    separator. An off-by-one here does not raise; it silently makes every
    downstream span check compare the wrong characters.
    """
    mismatches = []
    for doc in bc5cdr_dev:
        for m in doc.mentions:
            if doc.full_text[m.start:m.end] != m.text:
                mismatches.append((doc.pmid, m.start, m.end, m.text,
                                   doc.full_text[m.start:m.end]))
    assert not mismatches, (
        f"{len(mismatches)} offset mismatches, first 3: {mismatches[:3]}")


def test_full_text_is_title_space_abstract(bc5cdr_dev):
    for doc in bc5cdr_dev[:50]:
        if doc.abstract:
            assert doc.full_text == f"{doc.title} {doc.abstract}"
        else:
            assert doc.full_text == doc.title


def test_dev_split_shape(bc5cdr_dev):
    """Pins the corpus counts of the dev split."""
    assert len(bc5cdr_dev) == 500
    mentions = [m for d in bc5cdr_dev for m in d.mentions]
    assert len(mentions) == 9591
    assert sum(1 for m in mentions if m.mesh_id is None) == 59
    assert sum(1 for m in mentions if m.is_compound) == 100


def test_entity_types_are_the_two_expected(bc5cdr_dev):
    types = {m.entity_type for d in bc5cdr_dev for m in d.mentions}
    assert types == {"Chemical", "Disease"}


def test_no_mention_has_inverted_or_empty_span(bc5cdr_dev):
    for doc in bc5cdr_dev:
        for m in doc.mentions:
            assert 0 <= m.start < m.end <= len(doc.full_text)


def test_relations_reference_only_mesh_ids(bc5cdr_dev):
    for doc in bc5cdr_dev[:100]:
        for r in doc.relations:
            assert r.chemical_mesh_id
            assert r.disease_mesh_id
