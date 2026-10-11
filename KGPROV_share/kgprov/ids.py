"""
Content-addressed identifiers.

Every primary key in this system is a truncated hash of the content that
defines the record, never a positional counter. Three properties follow,
and each of them is load-bearing for a competency question:

  * Stable across re-extraction, so a mention keeps its identifier when
    the pipeline is re-run. Comparing the same mention across two run
    graphs needs this; CQ4 counts the run bundles such a comparison
    would join.
  * Independent of iteration order, so a set traversal cannot permute
    identifiers between runs.
  * Deterministic for unmappable entities, so the same ungrounded term
    converges on one provisional node across documents rather than
    spawning a new one each time. CQ7 looks for a later revision of
    exactly these nodes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_ID_LEN = 12
_WS = re.compile(r"\s+")
_TERMINAL_PUNCT = re.compile(r"[\s.,;:!?]+$")


def norm(s: str) -> str:
    """The single shared normalizer.

    NFKC, casefold, collapse internal whitespace, strip surrounding
    whitespace and terminal punctuation. Used for deduplication keys and
    for identifier hashing. Without one shared function, "TP53", "TP53 "
    and "tp53" enter grounding as three independent jobs with three
    chances to disagree.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _WS.sub(" ", s).strip()
    s = _TERMINAL_PUNCT.sub("", s)
    return s


def _digest(*parts: object) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:_ID_LEN]


def entity_id(doc_id: str, chunk_id: int, start: int, end: int, raw_span: str) -> str:
    return "E_" + _digest(doc_id, chunk_id, start, end, raw_span)


def relation_id(doc_id: str, chunk_id: int, subj_entity_id: str,
                obj_entity_id: str, raw_action_phrase: str) -> str:
    return "REL_" + _digest(doc_id, chunk_id, subj_entity_id,
                            obj_entity_id, raw_action_phrase)


def unmapped_id(resolved_string: str, semantic_category: str) -> str:
    """Provisional identifier for an entity the ontology cannot ground.

    Derived from the normalized string, so the same unmappable term
    yields the same node in every run and every document.
    """
    return "UNMAPPED:" + _digest(norm(resolved_string), semantic_category)


def dedup_key(resolved_string: str, semantic_category: str,
              section_id: str | None = None) -> tuple:
    """Grounding deduplication key: (norm(resolved_string), category).

    The pipeline does not call this function: phase2_ground builds the
    same pair inline and deduplicates within each document. section_id,
    when given, is appended to the key; no caller passes one. CQ8 counts
    the decision transfers that deduplication produces.
    """
    base = (norm(resolved_string), semantic_category)
    return base if section_id is None else base + (section_id,)


def space_id(kind: str, *discriminators: object) -> str:
    """Identifier for a constraint space.

    A space is identified by what produced it, not by its contents, so
    that two spaces with identical members but different derivations
    remain distinguishable. CQ3 depends on that distinction.
    """
    return "CS_" + _digest(kind, *discriminators)


def activity_id(phase: str, *discriminators: object) -> str:
    return "ACT_" + _digest(phase, *discriminators)


def content_hash(text: str) -> str:
    """Full-length hash, for prompt and configuration fingerprints."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
