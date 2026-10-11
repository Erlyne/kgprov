"""The four pipeline phases.

Each phase is a plain function that consumes and produces plain
dataclasses. Provenance is emitted through Recorder; LLM calls go
through BedrockClient. The phases write no tables or graphs; runner.py
wires everything together and writes those. Two caches are written
while the phases run: the response log, by BedrockClient, and the Tier 2
escalation cache, by EscalationCache.put in _run_tier2_agent.

Naming: the Phase 1 prompt and schema and the Phase 3 entity list use
uppercase CHEMICAL / DISEASE (schema enum values). The Tier 1 and Tier 2
prompts, the Phase 4 bucket keys, Mention.semantic_category and the
BC5CDR + MeSH internals use title-case Chemical / Disease. _from_uc()
and _to_uc() bridge the two.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from kgprov import (
    KGP, PROV, Recorder, ConstraintSpace, SpaceMember, ids, iri,
    ENFORCE_DECODE_TIME,
)
from kgprov.llm import BedrockClient, SchemaValidationError
from .chunking import Chunk, chunk_document, extract_abbreviations
from .loaders import Document, GoldMention, MeshOntology
from .registry import (
    EscalationCache, RegistryStats, is_wellformed, resolve_live,
    search_biomedical_web,
)
from .retrieval import (
    MeshIndex, RetrievalParams, RetrievalResult, retrieve,
)
from .sampling import ArmSampler


# ---------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------

@dataclass
class Mention:
    entity_id: str
    doc_id: str
    chunk_id: int
    start_char: int             # chunk-relative
    end_char: int
    doc_start_char: int
    raw_span: str
    resolved_string: str
    resolution_type: str        # NONE / ABBREVIATION / COREFERENCE / GENERIC
    semantic_category: str      # Chemical / Disease
    # Populated by Phase 2:
    cui: str | None = None
    mapping_tier: str | None = None       # LOCAL / WEB / NONE
    mapping_confidence: str | None = None # HIGH / MEDIUM / LOW
    candidate_score: float | None = None


@dataclass
class RejectedSpan:
    doc_id: str
    chunk_id: int
    raw_span: str
    reason: str


def _rests_on_provisional(*cuis: str) -> bool:
    """Does this row depend on an identifier the ontology could not confirm?

    True when either endpoint is a provisional ``UNMAPPED:`` identifier.
    Called from ``__post_init__`` rather than at each construction site,
    so ``is_provisional`` cannot be left unset.
    """
    return any(str(c).startswith("UNMAPPED:") for c in cuis)


@dataclass
class Relationship:
    relation_id: str
    doc_id: str
    chunk_id: int
    subject_entity_id: str
    object_entity_id: str
    subject_cui: str
    object_cui: str
    raw_action_phrase: str
    action_start_char: int
    action_end_char: int
    polarity: str
    modality: str
    conditions: dict
    is_provisional: bool = False

    def __post_init__(self) -> None:
        self.is_provisional = _rests_on_provisional(
            self.subject_cui, self.object_cui)


@dataclass
class KGEdge:
    subject_cui: str
    normalized_predicate: str
    object_cui: str
    polarity: str
    modality: str
    conditions: dict
    negated: bool
    evidence_count: int
    supporting_relation_ids: list[str]
    is_provisional: bool = False

    def __post_init__(self) -> None:
        self.is_provisional = _rests_on_provisional(
            self.subject_cui, self.object_cui)


#: Tier 1 resolves to exactly one of these. SELECTED and DECLINED are
#: model decisions; NOT_CALLED is the absence of one (empty candidate
#: pool, so no model was asked). They are never summed.
TIER1_SELECTED = "SELECTED"
TIER1_DECLINED = "DECLINED"
TIER1_NOT_CALLED = "NOT_CALLED"


@dataclass
class Phase2ArmDecision:
    """One arm's Tier 1 output for one dedup key."""
    dedup_key: tuple[str, str]
    arm: str                        # e.g. "weak_arm" or "strong_arm"
    mapped_id: str | None           # descriptor UI, or None
    confidence: str                 # HIGH / MEDIUM / LOW
    candidate_ids: list[str]        # the top-k pool the arm saw
    escalated: bool                 # True → Tier 2 was invoked
    outcome: str = TIER1_NOT_CALLED  # SELECTED | DECLINED | NOT_CALLED
    #: Tier 1 met the escalation trigger (NO_MATCH, or any id at LOW).
    #: Distinct from `escalated`, which says the unbounded cell actually
    #: ran: an arm can run bounded-only, and then the two differ. Keeping
    #: both means the unbounded cell's cost can be sized from a bounded
    #: run without inferring it from an absence.
    escalation_triggered: bool = False
    #: Every mention this grounding job decided for. Recorded at dispatch
    #: so evaluation joins on identifiers rather than re-deriving the link
    #: by matching Phase 1's rewritten strings against gold surface forms.
    covered_entity_ids: list[str] = field(default_factory=list)
    #: Populated on every arm that escalated. Escalation runs per arm so
    #: the unbounded cells of the 2x2 are measured rather than assumed.
    tier2_tool_calls: int = 0
    tier2_termination: str = ""
    tier2_claim: str = ""
    tier2_verified: bool = False
    #: Served from the persistent escalation cache. That cache
    #: short-circuits before the LLM client, so the call is absent from the
    #: response log and from cost_usd_notional (unlike a response-log
    #: replay, which notional does price).
    tier2_from_cache: bool = False
    #: Per-check outcomes, so "exists but denotes something else" stays
    #: distinguishable from "does not exist".
    tier2_checks: str = ""


# ---------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------

ALLOWED_CATEGORIES = ["CHEMICAL", "DISEASE"]

# Category normalization (LLM enum <-> MeSH type map)
_UC_TO_TC = {"CHEMICAL": "Chemical", "DISEASE": "Disease"}
_TC_TO_UC = {v: k for k, v in _UC_TO_TC.items()}

OMEGA = {
    "Chemical-Disease": [
        {"id": "causes",         "def": "The chemical causes or induces the disease/condition"},
        {"id": "contributes_to", "def": "The chemical increases risk or contributes to the condition"},
        {"id": "treats",         "def": "The chemical is used to treat the condition"},
        {"id": "prevents",       "def": "The chemical prevents the condition"},
    ],
    "Disease-Chemical": [
        {"id": "is_treated_by",  "def": "The disease is treated by the chemical"},
        {"id": "is_caused_by",   "def": "The disease is caused by the chemical"},
    ],
}
_OMEGA_EXTRA = ["PRED:NO_INTERACTION", "PRED:GARBLED"]


def _artifact_hash(schema: dict) -> str:
    """Fingerprint of the tool schema actually sent to the provider.

    Recorded in place of the schema itself. The admissible values are
    already the space's qualified memberships, and Phase 2 compiles a fresh
    candidate enum per call, so storing the JSON would duplicate the
    largest part of the graph. A test asserts this hash equals the hash of
    the schema the provider received.
    """
    return ids.content_hash(json.dumps(schema, sort_keys=True))


def _from_uc(cat_uc: str) -> str:
    return _UC_TO_TC.get(cat_uc, cat_uc)


def _to_uc(cat_tc: str) -> str:
    return _TC_TO_UC.get(cat_tc, cat_tc.upper())


def _hash_prompt(text: str) -> str:
    return ids.content_hash(text)[:16]


# ---------------------------------------------------------------------
#  Phase 1 prompt
# ---------------------------------------------------------------------

PHASE1_SYSTEM = """You are a biomedical entity extraction engine.
Extract entities ONLY from the text inside <target_chunk>.

GROUNDING BOUNDARY
- Every "raw_span" must appear verbatim and contiguously inside <target_chunk>.
- <context> is reference material only. Never extract entities from it.

EXTRACTION RULES
1. Raw span: Copy exact characters from <target_chunk>. Do not correct
   spelling, expand, re-case, or re-order.
2. Offsets: Report "start_char" and "end_char" as 0-based indices into
   <target_chunk> such that target_chunk[start_char:end_char] == raw_span.
3. Coordination: For "BRCA1 and BRCA2 mutations", emit one entity per
   conjunct. Set raw_span to the contiguous conjunct ("BRCA1"), put the
   reconstructed form in resolved_string ("BRCA1 mutation").
4. Resolution: Use <context> to expand abbreviations. Record what you did
   in resolution_type: NONE (span is unambiguous, resolved_string must
   equal raw_span), ABBREVIATION (expanded), COREFERENCE, or GENERIC.
5. Category: Choose semantic_category from [CHEMICAL, DISEASE] only.
6. If unsure, set resolution_type=NONE and resolved_string=raw_span.

Return a single JSON object with an "entities" array. No prose."""


#: Phase 1's admissible categories: a static constraint space, supplied as
#: a parameter and recoverable from the plan.
SEMANTIC_CATEGORIES = ["CHEMICAL", "DISEASE"]
RESOLUTION_TYPES = ["NONE", "ABBREVIATION", "COREFERENCE", "GENERIC"]

#: The schemas passed to complete_constrained() (Phase 1, Tier 1, Phase 3,
#: Phase 4) declare their enumerations, so the admissible set is sent to
#: the provider and enforced at decode time, not only rendered into the
#: prompt and checked afterwards.
PHASE1_SCHEMA = {
    "type": "object",
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["raw_span", "start_char", "end_char",
                             "resolved_string", "resolution_type",
                             "semantic_category"],
                "properties": {
                    "raw_span": {"type": "string"},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                    "resolved_string": {"type": "string"},
                    "resolution_type": {"type": "string",
                                        "enum": RESOLUTION_TYPES},
                    "semantic_category": {"type": "string",
                                          "enum": SEMANTIC_CATEGORIES},
                },
            },
        },
    },
}


def _phase1_user(title: str, abbrevs: dict[str, str], chunk_text: str) -> str:
    abbr_str = "; ".join(f"{k} = {v}" for k, v in abbrevs.items()) or "(none)"
    return (
        f"<context>\n"
        f"Title: {title}\n"
        f"Abbreviations: {abbr_str}\n"
        f"</context>\n\n"
        f"<target_chunk>\n{chunk_text}\n</target_chunk>"
    )


def _validate_phase1_entities(chunk_text: str, raw_entities: list[dict],
                              doc_id: str, chunk: Chunk,
                              rejected_out: list[RejectedSpan],
                              duplicates_out: list[str] | None = None
                              ) -> list[Mention]:
    """Post-validate Phase 1 output.

    Rejects spans with an invalid category or text not found in the chunk,
    repairs offsets when the span occurs elsewhere in the chunk, and
    collapses repeats by content-addressed identifier. The model can list
    the same occurrence several times in one response. Two records with
    the same `entity_id` are the same mention by construction, so keeping
    both would inflate every count taken from the CSVs: the mention total,
    the join denominator, and the decision-transfer share.

    A repeat is not a rejection and takes no reason code. Collapsed
    identifiers are reported through `duplicates_out`. In the graph,
    identical content-addressed IRIs merge on write.
    """
    kept: list[Mention] = []
    seen: set[str] = set()
    for e in raw_entities:
        try:
            raw_span = str(e["raw_span"])
            start = int(e["start_char"])
            end = int(e["end_char"])
            resolved = str(e.get("resolved_string", raw_span))
            rtype = str(e.get("resolution_type", "NONE")).upper()
            cat_uc = str(e.get("semantic_category", "")).upper()
        except (KeyError, TypeError, ValueError):
            continue

        if cat_uc not in ALLOWED_CATEGORIES:
            rejected_out.append(RejectedSpan(
                doc_id, chunk.chunk_id, raw_span, "CATEGORY_INVALID"))
            continue

        # Verify verbatim; repair or reject.
        got = chunk_text[start:end] if 0 <= start < end <= len(chunk_text) else ""
        if got != raw_span:
            idx = chunk_text.find(raw_span)
            if idx < 0:
                rejected_out.append(RejectedSpan(
                    doc_id, chunk.chunk_id, raw_span, "SPAN_NOT_IN_CHUNK"))
                continue
            start, end = idx, idx + len(raw_span)

        # Enforce resolution_type=NONE => resolved_string == raw_span
        if rtype == "NONE" and resolved != raw_span:
            resolved = raw_span

        cat_tc = _from_uc(cat_uc)
        eid = ids.entity_id(doc_id, chunk.chunk_id, start, end, raw_span)
        if eid in seen:
            if duplicates_out is not None:
                duplicates_out.append(eid)
            continue
        seen.add(eid)
        kept.append(Mention(
            entity_id=eid,
            doc_id=doc_id,
            chunk_id=chunk.chunk_id,
            start_char=start,
            end_char=end,
            doc_start_char=chunk.doc_offset + start,
            raw_span=raw_span,
            resolved_string=resolved,
            resolution_type=rtype,
            semantic_category=cat_tc,
        ))
    return kept


def phase1_extract(doc: Document, chunks: list[Chunk], *,
                   llm: BedrockClient, model_id: str,
                   recorder: Recorder | None,
                   temperature: float = 0.0,
                   max_tokens: int = 2048,
                   duplicates_out: list[str] | None = None
                   ) -> tuple[list[Mention], list[RejectedSpan]]:
    """Extract entities from every chunk of one document.

    ``duplicates_out``, when supplied, accumulates the identifiers the model
    emitted more than once, so the collapse is reported rather than assumed.
    """
    abbrevs_full = extract_abbreviations(doc.full_text)
    all_mentions: list[Mention] = []
    rejected: list[RejectedSpan] = []

    for chunk in chunks:
        # Per-chunk abbrev filter: only those whose short form appears here.
        chunk_abbrevs = {k: v for k, v in abbrevs_full.items()
                         if k in chunk.text}
        user_msg = _phase1_user(doc.title, chunk_abbrevs, chunk.text)
        prompt_hash = _hash_prompt(PHASE1_SYSTEM + user_msg)

        cat_space = ConstraintSpace(
            kind="semantic_category_enum",
            members=[SpaceMember(value=c, rank=i+1)
                     for i, c in enumerate(ALLOWED_CATEGORIES)],
            derived=False,
            discriminators=("phase1_categories",),
        )

        try:
            resp = llm.complete_constrained(
                model_id=model_id, system=PHASE1_SYSTEM, user=user_msg,
                tool_name="submit_entities", tool_schema=PHASE1_SCHEMA,
                description="Submit the extracted entities.",
                temperature=temperature, max_tokens=max_tokens)
            raw_entities = resp.parsed.get("entities", []) if resp.parsed else []
        except (SchemaValidationError, Exception) as e:
            print(f"[phase1 ERROR] doc={doc.pmid} chunk={chunk.chunk_id} "
                  f"{type(e).__name__}: {e}", flush=True)
            rejected.append(RejectedSpan(
                doc.pmid, chunk.chunk_id, "", f"LLM_ERROR: {type(e).__name__}"))
            continue

        kept = _validate_phase1_entities(
            chunk.text, raw_entities, doc.pmid, chunk, rejected,
            duplicates_out=duplicates_out)
        all_mentions.extend(kept)

        if recorder is not None:
            # Register chunk entity + doc source; safe on repeated calls.
            recorder.entity(f"doc_{doc.pmid}", cls=KGP.SourceManuscript)
            chunk_ent_id = f"chunk_{doc.pmid}_{chunk.chunk_id}"
            recorder.entity(chunk_ent_id, cls=KGP.TargetChunk)
            recorder.primary_source(chunk_ent_id, f"doc_{doc.pmid}")

            with recorder.generation(
                "phase1", agent="agent_extraction",
                plan="plan_extraction", prompt_hash=prompt_hash,
                autonomy_level=1,
                discriminators=(doc.pmid, chunk.chunk_id)) as call:
                call.constrained_by(
                    cat_space, enforcement=ENFORCE_DECODE_TIME,
                    enforcement_artifact=_artifact_hash(PHASE1_SCHEMA))
                call.used(chunk_ent_id)
                for m in kept:
                    call.generated(
                        m.entity_id, cls=KGP.GroundedMention,
                        rawSpan=m.raw_span, resolvedString=m.resolved_string,
                        semanticCategory=m.semantic_category,
                        startChar=m.start_char, endChar=m.end_char,
                        resolutionType=m.resolution_type)
                    recorder.quoted_from(m.entity_id, chunk_ent_id)
            for rj in [r for r in rejected
                       if r.doc_id == doc.pmid and r.chunk_id == chunk.chunk_id]:
                # Emit an invalidation for each span or category rejection.
                if rj.reason.startswith("SPAN_NOT_IN_CHUNK"):
                    tmp_id = "E_reject_" + ids.content_hash(
                        rj.raw_span + str(chunk.chunk_id))[:12]
                    recorder.entity(tmp_id, cls=KGP.RawSpan, rawSpan=rj.raw_span)
                    recorder.invalidate(
                        tmp_id, KGP.SpanNotInChunk,
                        by_activity=ids.activity_id(
                            "phase1", doc.pmid, chunk.chunk_id))
                elif rj.reason.startswith("CATEGORY"):
                    tmp_id = "E_reject_" + ids.content_hash(
                        rj.raw_span + str(chunk.chunk_id))[:12]
                    recorder.entity(tmp_id, cls=KGP.RawSpan, rawSpan=rj.raw_span)
                    recorder.invalidate(
                        tmp_id, KGP.CategoryInvalid,
                        by_activity=ids.activity_id(
                            "phase1", doc.pmid, chunk.chunk_id))

    return all_mentions, rejected


# ---------------------------------------------------------------------
#  Phase 2 prompts and helpers
# ---------------------------------------------------------------------

PHASE2_TIER1_SYSTEM = """You are an ontology mapping evaluator.

Decide whether any candidate in <candidate_list> denotes exactly the same
biological concept as <target_entity>.

- Match only if it is the SAME concept, not related, broader, or narrower.
- If exactly one matches, return its ID.
- If none match, return "NO_MATCH".
- If multiple match, return the most specific and set confidence to MEDIUM.

Set confidence:
  HIGH   - exact name or synonym match, correct type
  MEDIUM - confident paraphrase or morphological variant
  LOW    - plausible but unverified; will be escalated

Return a single JSON object with keys mapped_id and confidence. No prose."""


CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]

#: Shape-only schema with no admissible set. No phase sends it.
PHASE2_SCHEMA_UNCONSTRAINED = {
    "type": "object",
    "required": ["mapped_id", "confidence"],
    "properties": {
        "mapped_id": {"type": "string"},
        "confidence": {"type": "string"},
    },
}


def _tier1_schema(candidates: list[dict]) -> dict:
    """Compile the candidate enum for one grounding call.

    This is the derived constraint space: the admissible set is not in the
    plan; it is built at runtime from the retrieval chain's output and is
    gone when the run ends. Declaring it in the schema makes an excluded
    descriptor unreachable rather than merely discouraged.
    """
    ids_ = [c["id"] for c in candidates]
    return {
        "type": "object",
        "required": ["mapped_id", "confidence"],
        "properties": {
            "mapped_id": {"type": "string", "enum": ids_ + ["NO_MATCH"]},
            "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        },
    }


def _tier1_user(query: str, category_tc: str,
                candidates: list[dict]) -> str:
    return (
        f"<target_entity>\n"
        f"String:   {query}\n"
        f"Category: {category_tc}\n"
        f"</target_entity>\n\n"
        f"<candidate_list>\n"
        f"{json.dumps(candidates, indent=2)}\n"
        f"</candidate_list>"
    )


def _candidate_pool(index: MeshIndex,
                    result: RetrievalResult,
                    max_synonyms: int = 4) -> list[dict]:
    pool: list[dict] = []
    for ui, _score in result.final:
        entry = None
        # Look up entry via preferred name + synonyms.
        pref = index.preferred_name(ui) or ui
        syns: list[str] = []
        for surf, u in index.surface_forms:
            if u == ui and surf != pref:
                if surf not in syns:
                    syns.append(surf)
                if len(syns) >= max_synonyms:
                    break
        types = sorted(index.types_by_ui.get(ui, set()))
        pool.append({
            "id": ui,
            "preferred_name": pref,
            "synonyms": syns,
            "entity_types": types,
        })
    return pool


def _run_tier1(arm: str, model_id: str, query: str, category_tc: str,
               candidates: list[dict], *,
               llm: BedrockClient, temperature: float, max_tokens: int
               ) -> tuple[str | None, str]:
    """Return (mapped_id_or_NO_MATCH, confidence)."""
    user_msg = _tier1_user(query, category_tc, candidates)
    try:
        # The candidate enum is compiled per call and enforced at decode
        # time: the tokens spelling a non-candidate are masked, so the
        # membership check below does not fail for that reason.
        resp = llm.complete_constrained(
            model_id=model_id, system=PHASE2_TIER1_SYSTEM, user=user_msg,
            tool_name="submit_mapping",
            tool_schema=_tier1_schema(candidates),
            description="Submit the mapped ontology identifier.",
            temperature=temperature, max_tokens=max_tokens)
        mapped = str(resp.parsed.get("mapped_id", "NO_MATCH"))
        conf = str(resp.parsed.get("confidence", "LOW")).upper()
        if conf not in {"HIGH", "MEDIUM", "LOW"}:
            conf = "LOW"
        # Validate mapped_id is one of the candidate UIs or NO_MATCH.
        valid = {c["id"] for c in candidates} | {"NO_MATCH"}
        if mapped not in valid:
            return "NO_MATCH", "LOW"
        return (mapped if mapped != "NO_MATCH" else None), conf
    except Exception as e:
        print(f"[phase2 {arm} ERROR] {type(e).__name__}: {e}", flush=True)
        return None, "LOW"


_PRIOR_NORM_INDEX: dict[int, dict[str, list[str]]] = {}


def _prior_index(prior: MeshOntology) -> dict[str, list[str]]:
    """Build (once, per process) a norm(surface) -> [ui] map."""
    key = id(prior)
    cached = _PRIOR_NORM_INDEX.get(key)
    if cached is not None:
        return cached
    m: dict[str, list[str]] = {}
    for surf, ui in prior.surface_forms:
        m.setdefault(ids.norm(surf), []).append(ui)
    _PRIOR_NORM_INDEX[key] = m
    return m


def _tier2_lookup(query: str, category_tc: str,
                  primary: MeshOntology,
                  prior: MeshOntology | None) -> tuple[str | None, str]:
    """Look the query up in the prior MeSH release.

    No code in the pipeline calls this: phase2_ground escalates through
    _run_tier2_agent, and tests/test_tier2_agent.py checks that it does
    not call this function.

    Returns (descriptor_ui_or_None, source_url). Uses a norm-indexed
    dict so this is O(1) per call, not O(|surface_forms|).
    """
    if prior is None:
        return None, "urn:none"
    idx = _prior_index(prior)
    for ui in idx.get(ids.norm(query), []):
        e = prior.entries.get(ui)
        if e and category_tc in e.entity_types:
            return ui, f"https://id.nlm.nih.gov/mesh/{ui}"
    return None, "urn:mesh:not_found"


PHASE2_TIER2_SYSTEM = """The local ontology failed to map this entity.

String: {query}
Category: {category}

You have a live biomedical registry search tool. Use it to find the
official MeSH descriptor identifier for this entity, if one exists.

- Compose your own search terms. Try a different phrasing if the first
  returns nothing useful.
- Submit an identifier only if the registry result denotes the SAME
  concept, not merely a related, broader or narrower one.
- If the registry has no matching descriptor, submit the exact string
  UNMAPPED. A confident UNMAPPED is more useful than a guess.
- Never submit an identifier you have not seen in a registry result.
"""

_TIER2_SUBMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_cui": {
            "type": "string",
            "description": "The MeSH descriptor UI seen in a registry "
                           "result, or the exact string UNMAPPED.",
        },
        "source_url": {
            "type": "string",
            "description": "The registry URL the identifier was read from, "
                           "or an empty string if UNMAPPED.",
        },
    },
    "required": ["final_cui"],
}

_TIER2_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string",
                             "description": "Registry search terms."}},
    "required": ["query"],
}


def _run_tier2_agent(arm: str, model_id: str, query: str, category_tc: str,
                     *, llm: BedrockClient,
                     cache: "EscalationCache | None" = None,
                     stats: "RegistryStats | None" = None,
                     max_tool_calls: int = 6,
                     temperature: float = 0.0,
                     max_tokens: int = 2048) -> tuple:
    """Escalate to an unbounded agent over the live registry.

    Returns (claim_ui_or_None, source_url, trace) where trace carries the
    tool-call count and termination reason for the provenance record.

    This is autonomy level 4: no enumerable admissible set exists, so
    constraint capture does not apply and attestation is the only
    available provenance. The model decides whether to search, what to
    ask, whether to ask again, and what to submit; the controller decides
    only when to stop. The level is not written to the graph: the Tier 2
    nodes carry no kgp:autonomyLevel.
    """
    nq = ids.norm(query)
    if cache is not None:
        hit = cache.get(nq, category_tc, arm)
        if hit is not None:
            return (hit.get("claim_ui"), hit.get("source_url", "urn:cached"),
                    {**hit.get("trace", {}), "from_cache": True})

    def executor(name: str, args: dict) -> str:
        if name != "search_biomedical_web":
            return f"Unknown tool {name}."
        return search_biomedical_web(str(args.get("query", "")), stats=stats)

    run = llm.run_agent(
        model_id=model_id,
        system=PHASE2_TIER2_SYSTEM.format(query=query, category=category_tc),
        user="Find the official identifier for this entity.",
        tools=[BedrockClient.tool_spec(
            "search_biomedical_web",
            "Search the live MeSH registry for a descriptor.",
            _TIER2_SEARCH_SCHEMA)],
        executor=executor,
        submit_tool_name="submit_tier2_result",
        submit_schema=_TIER2_SUBMIT_SCHEMA,
        submit_description="Submit the discovered identifier or UNMAPPED.",
        max_tool_calls=max_tool_calls,
        temperature=temperature, max_tokens=max_tokens)

    sub = run.submitted or {}
    raw = str(sub.get("final_cui", "UNMAPPED")).strip()
    claim = None if raw.upper() in {"UNMAPPED", ""} else raw
    url = str(sub.get("source_url") or "")
    if claim and not url:
        url = f"https://id.nlm.nih.gov/mesh/{claim}"
    if not url:
        url = "urn:mesh:not_found"

    trace = {
        "n_tool_calls": run.n_tool_calls,
        "n_model_turns": run.n_model_turns,
        "tool_queries": [c["input"].get("query") for c in run.tool_calls],
        # prov:endedAtTime says an activity ended; nothing in PROV says
        # whether the agent concluded or the controller cut it off, and
        # the two have different remedies.
        "termination_reason": ("BUDGET_EXHAUSTED" if run.budget_exhausted
                              else "MODEL_CONCLUDED"),
        "from_cache": False,
    }
    if cache is not None:
        cache.put(nq, category_tc, arm,
                  {"claim_ui": claim, "source_url": url, "trace": trace})
    return claim, url, trace


def _verify_tier2_live(candidate_ui: str, primary: MeshOntology,
                       query: str, category_tc: str,
                       *, stats: "RegistryStats | None" = None) -> tuple:
    """Three admission checks. Returns (passed, per-check outcomes).

    An identifier returned by a web agent is a claim about an external
    registry, and MeSH identifiers are format-regular enough that a
    well-formed string is no evidence of existence -- D999999 is as
    fluent as D008687. An identifier present in the local table passes
    the existence check; one absent locally is looked up in the live
    registry.
    """
    checks = {"EXISTENCE": None, "LEXICAL": None, "TYPE": None}

    if not is_wellformed(candidate_ui):
        checks["EXISTENCE"] = False
        return False, checks

    entry = primary.entries.get(candidate_ui)
    live = None
    if entry is not None:
        pref, syns, types = (entry.preferred_name, entry.synonyms,
                             entry.entity_types)
        checks["EXISTENCE"] = True
    else:
        live = resolve_live(candidate_ui, stats=stats)
        if live is None:
            # Absent from both local and live: fabricated, or the
            # registry was unreachable. Either way not admissible.
            checks["EXISTENCE"] = False
            return False, checks
        pref, syns, types = live.preferred_name, live.synonyms, None
        checks["EXISTENCE"] = True

    nq = ids.norm(query)
    q_tokens, p_tokens = set(nq.split()), set(ids.norm(pref).split())
    jacc = (len(q_tokens & p_tokens) / len(q_tokens | p_tokens)
            if q_tokens and p_tokens else 0.0)
    checks["LEXICAL"] = bool(jacc >= 0.4
                             or any(ids.norm(s) == nq for s in syns))

    # Type agreement is only checkable where the local tree data exists;
    # unknown is not the same as incompatible.
    checks["TYPE"] = True if types is None else (category_tc in types)

    return all(checks.values()), checks


def _verify_tier2(candidate_ui: str, primary: MeshOntology,
                  query: str, category_tc: str) -> bool:
    """Local-only verification, without the live-registry fallback."""
    if candidate_ui not in primary.entries:
        return False
    e = primary.entries[candidate_ui]
    if category_tc not in e.entity_types:
        return False
    # Loose name similarity: overlap of normalized tokens.
    nq = ids.norm(query)
    np = ids.norm(e.preferred_name)
    q_tokens = set(nq.split())
    p_tokens = set(np.split())
    if not q_tokens or not p_tokens:
        return False
    jacc = len(q_tokens & p_tokens) / len(q_tokens | p_tokens)
    if jacc >= 0.4:
        return True
    # Or query is a listed synonym.
    for syn in e.synonyms:
        if ids.norm(syn) == nq:
            return True
    return False


# ---------------------------------------------------------------------
#  Phase 2 driver
# ---------------------------------------------------------------------

def phase2_ground(mentions: list[Mention], *,
                  index: MeshIndex, params: RetrievalParams,
                  ontology: MeshOntology,
                  ontology_prior: MeshOntology | None,
                  llm: BedrockClient,
                  tier1_arms: list[tuple[str, str]],
                  primary_arm: str,
                  recorder: Recorder | None,
                  temperature: float = 0.0,
                  max_tokens: int = 2048,
                  retrieval_cache: dict | None = None,
                  escalation_cache: EscalationCache | None = None,
                  registry_stats: RegistryStats | None = None,
                  tier2_max_tool_calls: int = 6,
                  tier2_arms: list[str] | None = None,
                  sampler: ArmSampler | None = None,
                  compact_narrowing: bool = True
                  ) -> tuple[list[Mention], list[Phase2ArmDecision]]:
    """Ground every mention.

    - Dedup by (norm(resolved_string), semantic_category).
    - Run retrieval ONCE per unique key.
    - Run Tier 1 for each arm over the same candidate set.
    - Downstream mention.cui is set from `primary_arm`'s decision.
    - retrieval_cache: optional dict[(norm_text, cat_tc)] -> RetrievalResult
      to skip re-running retrieval across sessions.
    - tier2_arms: which arms run the unbounded cell. `None` means every
      Tier 1 arm (the full 2x2). Naming a subset runs the bounded cell for
      an arm without its unbounded one.
    - sampler: optional per-key sampling for the expensive arms
      (`pipeline.sampling`). An arm not admitted for a key produces no
      decision row for it, rather than a row carrying a fourth outcome
      value: `NOT_CALLED` means the candidate pool was empty, and
      "outside the sample" is not that. The admitted key set
      is written out by the runner so the sample is auditable rather than
      inferred from which rows are absent.
    """
    arm_names = [a for a, _ in tier1_arms]
    if tier2_arms is None:
        tier2_set = set(arm_names)
    else:
        tier2_set = set(tier2_arms)
        unknown = tier2_set - set(arm_names)
        if unknown:
            # Escalation is triggered by Tier 1 failure, so an arm with no
            # Tier 1 decision has nothing to escalate from.
            raise ValueError(
                f"tier2_arms not in tier1_arms: {sorted(unknown)}")

    if sampler is not None and sampler.samples(primary_arm):
        # The primary arm's decision is what stamps mention.cui, so
        # sampling it would leave mentions ungrounded rather than
        # ungrounded-by-measurement.
        raise ValueError(
            f"primary_arm {primary_arm!r} cannot be a sampled arm")

    # Bucket by dedup key.
    buckets: dict[tuple[str, str], list[Mention]] = {}
    for m in mentions:
        key = (ids.norm(m.resolved_string), m.semantic_category)
        buckets.setdefault(key, []).append(m)

    arm_decisions: list[Phase2ArmDecision] = []
    if retrieval_cache is None:
        retrieval_cache = {}

    for key, members in buckets.items():
        norm_q, cat_tc = key
        # Use the first mention's resolved string as the query (readability).
        query = members[0].resolved_string
        disc = (norm_q, cat_tc)

        result = retrieval_cache.get(key)
        if result is None:
            # Pass recorder so narrowing activities land in the graph:
            # CQ3 reads them to localize where a gold ID was lost.
            result = retrieve(index, query, cat_tc, params,
                              recorder=recorder, discriminators=disc,
                              compact_narrowing=compact_narrowing)
            retrieval_cache[key] = result

        candidates = _candidate_pool(index, result)
        cand_uis = [c["id"] for c in candidates]

        candidate_space = ConstraintSpace(
            kind="top_k",
            members=[SpaceMember(
                value=c["id"], rank=i+1,
                score=result.final[i][1] if i < len(result.final) else None,
                label=c["preferred_name"])
                for i, c in enumerate(candidates)],
            derived=True,
            discriminators=("phase2_topk",) + disc,
        )

        arm_results: dict[str, tuple[str | None, str]] = {}

        # Per-arm Tier 1
        for arm_name, model_id in tier1_arms:
            if sampler is not None and not sampler.admits(arm_name, key):
                # Outside this arm's sample. No call, no row, no space --
                # and no Tier 2 either, since escalation needs a Tier 1
                # decision to escalate from, which is what puts both of an
                # arm's cells on one key set.
                continue
            if not candidates:
                # The pool is empty (type filter or score floor), so no
                # model is asked.
                # This is not a declination: nothing expressed an opinion.
                mid, conf, outcome = None, "LOW", TIER1_NOT_CALLED
            else:
                mid, conf = _run_tier1(
                    arm_name, model_id, query, cat_tc, candidates,
                    llm=llm, temperature=temperature, max_tokens=max_tokens)
                outcome = (TIER1_SELECTED if mid is not None
                           else TIER1_DECLINED)
            arm_results[arm_name] = (mid, conf)

            if recorder is not None and candidates:
                prompt_hash = _hash_prompt(
                    PHASE2_TIER1_SYSTEM + query + cat_tc +
                    json.dumps(cand_uis))
                # Mark the selected candidate for the space snapshot.
                sel_space = ConstraintSpace(
                    kind="top_k",
                    members=[SpaceMember(
                        value=c["id"], rank=i+1,
                        score=(result.final[i][1]
                               if i < len(result.final) else None),
                        selected=(mid is not None and c["id"] == mid),
                        label=c["preferred_name"])
                        for i, c in enumerate(candidates)],
                    derived=True,
                    discriminators=("phase2_topk", arm_name) + disc,
                )
                with recorder.generation(
                    f"phase2_tier1_{arm_name}",
                    agent="agent_evaluator",
                    plan=f"plan_tier1_{arm_name}",
                    prompt_hash=prompt_hash,
                    autonomy_level=3,
                    discriminators=("tier1", arm_name) + disc) as call:
                    call.constrained_by(
                        sel_space, enforcement=ENFORCE_DECODE_TIME,
                        enforcement_artifact=_artifact_hash(
                            _tier1_schema(candidates)))
                    for m in members[:1]:  # attribute to the first mention id
                        call.used(m.entity_id)
                    if mid is not None:
                        call.generated(
                            f"cui_{arm_name}_{mid}_{disc[0][:12]}_{disc[1]}",
                            cls=KGP.CanonicalConceptID,
                            cuiValue=mid, mappingTier="LOCAL",
                            mappingConfidence=conf,
                            candidateScore=(
                                result.final[cand_uis.index(mid)][1]
                                if mid in cand_uis else None),
                            arm=arm_name)

            triggered = (mid is None or conf == "LOW")
            arm_decisions.append(Phase2ArmDecision(
                dedup_key=key, arm=arm_name,
                mapped_id=mid, confidence=conf,
                candidate_ids=cand_uis,
                escalated=(triggered and arm_name in tier2_set),
                escalation_triggered=triggered,
                outcome=outcome,
                covered_entity_ids=[m.entity_id for m in members],
            ))

        # Primary-arm outcome drives downstream propagation.
        primary_mid, primary_conf = arm_results.get(primary_arm, (None, "LOW"))
        final_cui: str | None = primary_mid
        final_tier = "LOCAL" if primary_mid else None
        final_score = None
        if primary_mid and primary_mid in cand_uis:
            i = cand_uis.index(primary_mid)
            if i < len(result.final):
                final_score = result.final[i][1]

        # Tier 2 escalation: an unbounded agent over the live registry.
        # Autonomy level 4: no enumerable admissible set exists, so
        # constraint capture does not apply and attestation is the only
        # available provenance. Escalation runs per arm so that the
        # unbounded cells of the 2x2 are measured. Only the primary arm's
        # result propagates downstream; the others are recorded for the
        # comparison and change no graph content.
        tier2_by_arm: dict[str, dict] = {}
        for arm_name, arm_model in tier1_arms:
            if arm_name not in tier2_set:
                continue
            if arm_name not in arm_results:
                # Tier 1 never ran for this arm on this key, so there is no
                # failure to escalate from. Defaulting to (None, "LOW")
                # here would escalate a key the arm was never asked about
                # and bill the unbounded cell for a sample it is not in.
                continue
            a_mid, a_conf = arm_results[arm_name]
            if a_mid is not None and a_conf != "LOW":
                continue
            try:
                a_claim, a_url, a_trace = _run_tier2_agent(
                    arm_name, arm_model, query, cat_tc,
                    llm=llm, cache=escalation_cache, stats=registry_stats,
                    max_tool_calls=tier2_max_tool_calls,
                    temperature=temperature, max_tokens=max_tokens)
            except Exception as e:                        # noqa: BLE001
                # A failed escalation is recorded as UNMAPPED with a stated
                # cause (TOOL_FAILURE) instead of aborting the run, the same
                # outcome the controller produces on budget exhaustion.
                # Phases 1 and 3 absorb a failed generation in the same way.
                print(f"[phase2 tier2 ERROR] arm={arm_name} "
                      f"query={query!r} {type(e).__name__}: {e}", flush=True)
                a_claim, a_url = None, "urn:tier2:error"
                a_trace = {
                    "n_tool_calls": 0, "n_model_turns": 0,
                    "tool_queries": [],
                    "termination_reason": "TOOL_FAILURE",
                    "from_cache": False,
                }
            a_verified, a_checks = (False, {})
            if a_claim is not None:
                a_verified, a_checks = _verify_tier2_live(
                    a_claim, ontology, query, cat_tc, stats=registry_stats)
            tier2_by_arm[arm_name] = {
                "claim": a_claim, "url": a_url, "trace": a_trace,
                "verified": a_verified, "checks": a_checks,
            }
            # Attach the autonomy this arm actually exercised.
            for d in arm_decisions:
                if d.dedup_key == key and d.arm == arm_name:
                    d.tier2_tool_calls = a_trace.get("n_tool_calls", 0)
                    d.tier2_termination = a_trace.get("termination_reason", "")
                    d.tier2_claim = a_claim or ""
                    d.tier2_verified = a_verified
                    d.tier2_from_cache = bool(a_trace.get("from_cache"))
                    d.tier2_checks = "|".join(
                        f"{k}={'pass' if v else 'fail'}"
                        for k, v in a_checks.items() if v is not None)

        if primary_arm in tier2_by_arm:
            _t2 = tier2_by_arm[primary_arm]
            claim_ui, source_url = _t2["claim"], _t2["url"]
            t2_trace, verified, checks = (_t2["trace"], _t2["verified"],
                                          _t2["checks"])
            if recorder is not None:
                # Attach the attestation to the primary mention.
                first_eid = members[0].entity_id
                claim_local_id = f"claim_{first_eid}"
                recorder.entity(claim_local_id, cls=KGP.CanonicalConceptID,
                                cuiValue=(claim_ui or "UNMAPPED"),
                                mappingTier="WEB")

                # The unbounded step and the autonomy it exercised. Each
                # tool call is an activity whose *arguments* the model
                # chose: the tool is deterministic, the query is not.
                # No nToolCalls / nModelTurns: those are derivable by
                # counting ToolInvocation instances, and storing a derived
                # aggregate alongside the facts it summarises invites the
                # two to disagree.
                unbounded_id = f"unbounded_{first_eid}"
                recorder.entity(
                    unbounded_id, cls=KGP.UnboundedGeneration,
                    callBudget=tier2_max_tool_calls,
                    # prov:endedAtTime says an activity ended; PROV has no
                    # predicate for whether the agent concluded or the
                    # controller cut it off, and the remedies differ.
                    terminationReason=t2_trace.get("termination_reason", ""))
                for i, q in enumerate(t2_trace.get("tool_queries") or [], 1):
                    if q:
                        tc_id = f"toolcall_{first_eid}_{i}"
                        recorder.entity(
                            tc_id, cls=KGP.ToolInvocation,
                            toolName="search_biomedical_web",
                            # The tool is a deterministic HTTP call; the
                            # query is the generative act. Recording only
                            # that a search happened loses the agency.
                            toolArguments=str(q), invocationOrdinal=i)
                        recorder.g.add((iri(tc_id), PROV.wasInformedBy,
                                        iri(unbounded_id)))

                with recorder.attestation(
                    agent="agent_escalation",
                    subject=claim_local_id,
                    discriminators=("tier2",) + disc) as att:
                    att.against(source_url,
                                specialization_of="mesh_registry")
                    # Three checks recorded separately: an identifier that
                    # exists but denotes something else is a different
                    # failure from one that does not exist at all.
                    for kind, passed in checks.items():
                        if passed is not None:
                            recorder.entity(
                                f"check_{first_eid}_{kind}",
                                cls=KGP.VerificationCheck,
                                checkKind=kind, checkOutcome=bool(passed))
                    if verified:
                        att.outcome(KGP.Confirmed, repeatable=False)
                    elif claim_ui is not None:
                        # Submitted and not admissible. No separate
                        # claimWasFabricated term: a fabrication is Refuted
                        # with the EXISTENCE check failed. An unreachable
                        # registry produces the same pair.
                        att.outcome(KGP.Refuted, repeatable=False)
                    else:
                        att.outcome(KGP.Unverifiable, repeatable=False)
            if verified:
                final_cui = claim_ui
                final_tier = "WEB"

        if final_cui is None:
            final_cui = ids.unmapped_id(query, cat_tc)
            final_tier = "NONE"
            primary_conf = "LOW"
            if recorder is not None:
                recorder.entity(
                    f"cui_{final_cui}_{members[0].entity_id}",
                    cls=KGP.ProvisionalConceptID,
                    cuiValue=final_cui, mappingTier="NONE")

        # Stamp result onto every mention in this dedup bucket.
        first_eid = members[0].entity_id
        for i, m in enumerate(members):
            m.cui = final_cui
            m.mapping_tier = final_tier
            m.mapping_confidence = primary_conf
            m.candidate_score = final_score
            if i > 0 and recorder is not None:
                recorder.decision_transfer(
                    m.entity_id, first_eid,
                    dedup_key=f"dedup_{norm_q}_{cat_tc}")

    return mentions, arm_decisions


# ---------------------------------------------------------------------
#  Phase 3
# ---------------------------------------------------------------------

PHASE3_SYSTEM = """You are a biomedical relationship extraction engine.

Identify interactions between entities in <window_entities> that are stated
in <window_text>.

RULES
1. Participants: Output exact CUIs from the provided list as subject_cui
   and object_cui. Assign by grammatical role.
2. Action phrase: Copy the author's exact verbatim phrase.
3. Polarity: NEGATIVE when the text states the interaction did NOT occur.
   Otherwise POSITIVE.
4. Modality: ASSERTED for reported results. HYPOTHESIZED for hedged.
   PROPOSED for future/cited work.
5. Conditions: Populate each field only from explicit text. Use null for
   unstated. Do not infer species from cell line.
6. Offsets: action_start_char and action_end_char as 0-based into window_text.
7. Only extract interactions actually stated. Empty array is valid.

Return a single JSON object with an "extracted_relationships" array."""


POLARITIES = ["POSITIVE", "NEGATIVE"]
MODALITIES = ["ASSERTED", "HYPOTHESIZED", "PROPOSED"]


def _phase3_schema(window_cuis: list[str]) -> dict:
    """Compile the identifier enum for one extraction window.

    Derived from Phase 2's output, which is what makes derived spaces
    *chain*: an error upstream propagates here as a corrupted admissible
    set rather than as bad data, so the correct relation becomes
    unreachable rather than merely unselected.
    """
    cui_enum = {"type": "string", "enum": sorted(set(window_cuis))}
    return {
        "type": "object",
        "required": ["extracted_relationships"],
        "properties": {
            "extracted_relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["subject_cui", "object_cui",
                                 "raw_action_phrase", "action_start_char",
                                 "action_end_char", "polarity", "modality"],
                    "properties": {
                        "subject_cui": cui_enum,
                        "object_cui": cui_enum,
                        "raw_action_phrase": {"type": "string"},
                        "action_start_char": {"type": "integer"},
                        "action_end_char": {"type": "integer"},
                        # Enumerated so an unparseable polarity cannot be
                        # produced. Coercing one to POSITIVE would silently
                        # turn a negation into an assertion.
                        "polarity": {"type": "string", "enum": POLARITIES},
                        "modality": {"type": "string", "enum": MODALITIES},
                    },
                },
            },
        },
    }


def _phase3_user(window_text: str, entities: list[Mention]) -> str:
    ent_json = [{
        "entity_id": m.entity_id, "cui": m.cui,
        "semantic_category": _to_uc(m.semantic_category),
        "resolved_string": m.resolved_string,
    } for m in entities]
    return (
        f"<window_text>\n{window_text}\n</window_text>\n\n"
        f"<window_entities>\n{json.dumps(ent_json, indent=2)}\n</window_entities>"
    )


def _windows(chunks: list[Chunk], w: int = 1):
    """Yield (focus_chunk_id, window_chunks, window_text_offset_base)."""
    for i, focus in enumerate(chunks):
        lo = max(0, i - w)
        hi = min(len(chunks), i + w + 1)
        window = chunks[lo:hi]
        window_text = " ".join(c.text for c in window)
        yield focus.chunk_id, window, window_text


def _reject_relation(recorder, doc: Document, focus_id: int,
                     phrase: str, reason) -> None:
    """Record a Phase-3 row that was extracted but never reached the graph.

    Without this the row vanishes on a bare `continue` and CQ9
    under-reports. The provisional entity is content-addressed on the
    offending phrase so the same bad extraction converges on one node
    across runs rather than spawning a fresh one each time.
    """
    if recorder is None:
        return
    tmp_id = "REL_reject_" + ids.content_hash(
        f"{doc.pmid}|{focus_id}|{phrase}")[:12]
    recorder.entity(tmp_id, cls=KGP.VerbatimAction, rawActionPhrase=phrase)
    recorder.invalidate(
        tmp_id, reason,
        by_activity=ids.activity_id("phase3", doc.pmid, focus_id))


def phase3_extract_relations(doc: Document, chunks: list[Chunk],
                             mentions: list[Mention], *,
                             llm: BedrockClient, model_id: str,
                             recorder: Recorder | None,
                             w: int = 1,
                             temperature: float = 0.0,
                             max_tokens: int = 2048
                             ) -> list[Relationship]:
    relations: list[Relationship] = []
    seen_keys: set[tuple] = set()

    for focus_id, window, window_text in _windows(chunks, w):
        window_cids = {c.chunk_id for c in window}
        window_mentions = [m for m in mentions if m.chunk_id in window_cids]
        distinct_cuis = {m.cui for m in window_mentions if m.cui}
        if len(distinct_cuis) < 2:
            continue

        user_msg = _phase3_user(window_text, window_mentions)
        prompt_hash = _hash_prompt(PHASE3_SYSTEM + user_msg)

        cui_space = ConstraintSpace(
            kind="window_cuis",
            members=[SpaceMember(value=c, rank=i+1)
                     for i, c in enumerate(sorted(distinct_cuis))],
            derived=True,
            discriminators=(doc.pmid, focus_id),
        )

        try:
            resp = llm.complete_constrained(
                model_id=model_id, system=PHASE3_SYSTEM, user=user_msg,
                tool_name="submit_relationships",
                # Enum compiled from THIS window's identifiers, so the
                # space is derived from Phase 2's output rather than given.
                tool_schema=_phase3_schema(sorted(distinct_cuis)),
                description="Submit the extracted relationships.",
                temperature=temperature, max_tokens=max_tokens)
            raw = resp.parsed.get("extracted_relationships", []) if resp.parsed else []
        except Exception as e:
            print(f"[phase3 ERROR] doc={doc.pmid} focus={focus_id} "
                  f"{type(e).__name__}: {e}", flush=True)
            raw = []

        # Validate + record. Accept both raw_action_phrase and the
        # commonly-drifted alias "action_phrase"; reconstruct conditions
        # from top-level fields when the nested object is absent.
        emitted: list[Relationship] = []
        _COND_KEYS = {"species", "cell_line", "cell_type", "organ",
                      "tissue", "dose", "route", "assay_context", "other"}
        for r in raw:
            try:
                scui = str(r["subject_cui"])
                ocui = str(r["object_cui"])
                phrase = str(r.get("raw_action_phrase")
                             or r.get("action_phrase") or "")
                if not phrase:
                    continue
                a_start = int(r["action_start_char"])
                a_end = int(r["action_end_char"])
                polarity = str(r.get("polarity", "POSITIVE")).upper()
                modality = str(r.get("modality", "ASSERTED")).upper()
                conditions = r.get("conditions")
                if not isinstance(conditions, dict):
                    conditions = {k: r[k] for k in _COND_KEYS
                                  if k in r and r[k] is not None}
            except (KeyError, TypeError, ValueError):
                continue
            # CUIs must be in window
            if scui not in distinct_cuis or ocui not in distinct_cuis:
                continue
            if scui == ocui:
                continue
            # Verify action phrase verbatim. A phrase not found in the
            # window is recorded as a rejection (kgp:ActionNotVerbatim),
            # not silently dropped: CQ9 asks what was extracted and never
            # reached the graph.
            got = window_text[a_start:a_end] \
                if 0 <= a_start < a_end <= len(window_text) else ""
            if got != phrase:
                idx = window_text.find(phrase)
                if idx < 0:
                    _reject_relation(recorder, doc, focus_id, phrase,
                                     KGP.ActionNotVerbatim)
                    continue
                a_start, a_end = idx, idx + len(phrase)

            # A polarity outside the enum is rejected (kgp:PolarityInvalid)
            # rather than coerced to POSITIVE, which would turn "did not
            # inhibit" into an assertion of inhibition, with full
            # provenance.
            if polarity not in {"POSITIVE", "NEGATIVE"}:
                _reject_relation(recorder, doc, focus_id, phrase,
                                 KGP.PolarityInvalid)
                continue

            # Modality is an epistemic hedge, not a truth-value flip:
            # defaulting an unrecognised value to ASSERTED cannot invert
            # what the source claims, so it remains a default.
            if modality not in {"ASSERTED", "HYPOTHESIZED", "PROPOSED"}:
                modality = "ASSERTED"

            # Look up subject/object entity ids from window mentions.
            subj_ent = next((m for m in window_mentions if m.cui == scui), None)
            obj_ent = next((m for m in window_mentions if m.cui == ocui), None)
            if not subj_ent or not obj_ent:
                continue

            rid = ids.relation_id(doc.pmid, focus_id,
                                  subj_ent.entity_id, obj_ent.entity_id,
                                  phrase)
            dedup = (scui, ocui, ids.norm(phrase), polarity)
            if dedup in seen_keys:
                continue
            seen_keys.add(dedup)

            rel = Relationship(
                relation_id=rid, doc_id=doc.pmid, chunk_id=focus_id,
                subject_entity_id=subj_ent.entity_id,
                object_entity_id=obj_ent.entity_id,
                subject_cui=scui, object_cui=ocui,
                raw_action_phrase=phrase,
                action_start_char=a_start, action_end_char=a_end,
                polarity=polarity, modality=modality,
                conditions=conditions if isinstance(conditions, dict) else {},
            )
            emitted.append(rel)
            relations.append(rel)

        if recorder is not None:
            with recorder.generation(
                "phase3", agent="agent_relation",
                plan="plan_relation", prompt_hash=prompt_hash,
                autonomy_level=2,
                discriminators=(doc.pmid, focus_id)) as call:
                call.constrained_by(
                    cui_space, enforcement=ENFORCE_DECODE_TIME,
                    enforcement_artifact=_artifact_hash(
                        _phase3_schema(sorted(distinct_cuis))))
                for rel in emitted:
                    call.generated(
                        rel.relation_id, cls=KGP.KnowledgeGraphEdge,
                        subjectCui=rel.subject_cui,
                        objectCui=rel.object_cui,
                        rawActionPhrase=rel.raw_action_phrase,
                        polarity=rel.polarity,
                        modality=rel.modality)

    return relations


# ---------------------------------------------------------------------
#  Phase 4
# ---------------------------------------------------------------------

PHASE4_SYSTEM = """You are a database normalization engine.

For each item in <extraction_batch>, map the author's verbatim action
phrase to one predicate from <allowed_predicates>.

Both entities share the same type pair. <allowed_predicates> lists every
valid predicate with its definition. Choose by definition, not string
similarity.

RULES
1. Polarity is already determined. Ignore negation when choosing: "did not
   inhibit" maps to the same predicate as "inhibited".
2. Ambiguity: choose the more general predicate if its definition holds.
3. Escape hatches:
   - PRED:NO_INTERACTION - phrase describes no biological interaction
   - PRED:GARBLED - extraction noise, truncated, not a verb phrase
4. Return exactly one object per input item, echoing its "idx".

Return a single JSON object with a "normalized_edges" array."""


#: Escape codes the model may return instead of a predicate. They are part
#: of the admissible set so the model can decline instead of inventing a
#: predicate.
PREDICATE_ESCAPES = ["PRED:NO_INTERACTION", "PRED:GARBLED"]


def _phase4_schema(allowed_predicates: list[str]) -> dict:
    """Compile the predicate enum for one type-pair bucket.

    Derived from Phase 2's output via the ordered type pair. The predicate
    set is small on this corpus.
    """
    return {
        "type": "object",
        "required": ["normalized_edges"],
        "properties": {
            "normalized_edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["idx", "normalized_predicate"],
                    "properties": {
                        "idx": {"type": "integer"},
                        "normalized_predicate": {
                            "type": "string",
                            "enum": list(allowed_predicates)
                                    + PREDICATE_ESCAPES},
                    },
                },
            },
        },
    }


def _phase4_user(items: list[dict], allowed: list[dict]) -> str:
    return (
        f"<extraction_batch>\n{json.dumps(items, indent=2)}\n</extraction_batch>\n\n"
        f"<allowed_predicates>\n{json.dumps(allowed, indent=2)}\n</allowed_predicates>"
    )


def _bucket_key(subj_cat: str, obj_cat: str) -> str:
    return f"{subj_cat}-{obj_cat}"


def _lookup_cat(cui: str, mentions_by_cui: dict[str, str]) -> str | None:
    return mentions_by_cui.get(cui)


def phase4_normalize(relationships: list[Relationship],
                     mentions: list[Mention], *,
                     llm: BedrockClient, model_id: str,
                     recorder: Recorder | None,
                     batch_size: int = 50,
                     temperature: float = 0.0,
                     max_tokens: int = 2048) -> list[KGEdge]:
    """Normalize action phrases → predicates, bucket-by-bucket."""
    if not relationships:
        return []
    cat_by_cui: dict[str, str] = {}
    for m in mentions:
        if m.cui and m.cui not in cat_by_cui:
            cat_by_cui[m.cui] = m.semantic_category

    buckets: dict[str, list[tuple[int, Relationship, str]]] = {}
    unschemad: list[Relationship] = []
    for idx, rel in enumerate(relationships):
        scat = cat_by_cui.get(rel.subject_cui)
        ocat = cat_by_cui.get(rel.object_cui)
        if not scat or not ocat:
            unschemad.append(rel)
            continue
        key = _bucket_key(scat, ocat)
        allowed = OMEGA.get(key)
        # Try reverse direction only if forward missing.
        if allowed is None:
            rev = _bucket_key(ocat, scat)
            if OMEGA.get(rev) is not None:
                key = rev
                # Swap subj/obj for the LLM (schema-forward direction).
                rel_flipped = Relationship(
                    relation_id=rel.relation_id, doc_id=rel.doc_id,
                    chunk_id=rel.chunk_id,
                    subject_entity_id=rel.object_entity_id,
                    object_entity_id=rel.subject_entity_id,
                    subject_cui=rel.object_cui, object_cui=rel.subject_cui,
                    raw_action_phrase=rel.raw_action_phrase,
                    action_start_char=rel.action_start_char,
                    action_end_char=rel.action_end_char,
                    polarity=rel.polarity, modality=rel.modality,
                    conditions=rel.conditions)
                buckets.setdefault(key, []).append((idx, rel_flipped, "flipped"))
                continue
            unschemad.append(rel)
            continue
        buckets.setdefault(key, []).append((idx, rel, "direct"))

    # Emit invalidation for schema-less relations.
    if recorder is not None:
        for rel in unschemad:
            recorder.invalidate(rel.relation_id, KGP.NoConstraintSchema)

    predicate_by_idx: dict[int, str] = {}

    for bucket_key, items in buckets.items():
        allowed = OMEGA[bucket_key]
        allowed_predicates = [p["id"] for p in allowed] + _OMEGA_EXTRA

        pred_space = ConstraintSpace(
            kind="predicate_enum",
            members=[SpaceMember(value=p, rank=i+1)
                     for i, p in enumerate(allowed_predicates)],
            derived=False,
            discriminators=("phase4", bucket_key),
        )

        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_payload = [{
                "idx": local_idx,
                "raw_action_phrase": r.raw_action_phrase,
                "polarity": r.polarity,
                "evidence_snippet": r.raw_action_phrase[:200],
            } for local_idx, (_g_idx, r, _flag) in enumerate(batch)]

            user_msg = _phase4_user(batch_payload, allowed)
            prompt_hash = _hash_prompt(PHASE4_SYSTEM + user_msg)

            got: dict[int, str] = {}
            try:
                resp = llm.complete_constrained(
                    model_id=model_id, system=PHASE4_SYSTEM, user=user_msg,
                    tool_name="submit_predicates",
                    # The enum is Omega for this bucket plus the two escape
                    # codes, the same set recorded as pred_space.
                    tool_schema=_phase4_schema(
                        [p["id"] for p in allowed]),
                    description="Submit a normalized predicate per row.",
                    temperature=temperature, max_tokens=max_tokens)
                raw = resp.parsed.get("normalized_edges", []) if resp.parsed else []
                _extra_aliases = {
                    "NO_INTERACTION": "PRED:NO_INTERACTION",
                    "GARBLED": "PRED:GARBLED",
                }
                for r in raw:
                    try:
                        li = int(r["idx"])
                        pred = str(r.get("normalized_predicate")
                                   or r.get("predicate") or "")
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not pred:
                        continue
                    pred = _extra_aliases.get(pred, pred)
                    if pred in allowed_predicates and li not in got:
                        got[li] = pred
            except Exception as e:
                print(f"[phase4 ERROR] bucket={bucket_key} batch_start={i} "
                      f"{type(e).__name__}: {e}", flush=True)

            for local_idx, (g_idx, rel, _flag) in enumerate(batch):
                pred = got.get(local_idx)
                if pred is None:
                    pred = "REVIEW:DROPPED"
                predicate_by_idx[g_idx] = pred

            if recorder is not None:
                with recorder.generation(
                    "phase4", agent="agent_normalization",
                    plan="plan_normalize", prompt_hash=prompt_hash,
                    autonomy_level=2,
                    discriminators=("phase4", bucket_key, i)) as call:
                    call.constrained_by(
                        pred_space, enforcement=ENFORCE_DECODE_TIME,
                        enforcement_artifact=_artifact_hash(
                            _phase4_schema([p['id'] for p in allowed])))
                    for local_idx, (g_idx, rel, _flag) in enumerate(batch):
                        pred = predicate_by_idx.get(g_idx, "REVIEW:DROPPED")
                        if pred.startswith("PRED:") or pred == "REVIEW:DROPPED":
                            reason = (KGP.NoInteraction if pred == "PRED:NO_INTERACTION"
                                      else KGP.Garbled if pred == "PRED:GARBLED"
                                      else KGP.DroppedFromBatch)
                            recorder.invalidate(rel.relation_id, reason)
                        else:
                            call.generated(
                                f"pred_{rel.relation_id}",
                                cls=KGP.NormalizedPredicate,
                                normalizedPredicate=pred,
                                appliesTo=rel.relation_id)

    # Aggregate into edges (polarity guard: NEGATIVE → negated=True).
    agg: dict[tuple, KGEdge] = {}
    for idx, rel in enumerate(relationships):
        pred = predicate_by_idx.get(idx)
        if pred is None or pred.startswith("PRED:") or pred == "REVIEW:DROPPED":
            continue
        negated = (rel.polarity == "NEGATIVE")
        cond_key = json.dumps(rel.conditions, sort_keys=True)
        key = (rel.subject_cui, pred, rel.object_cui,
               rel.polarity, rel.modality, cond_key)
        if key not in agg:
            agg[key] = KGEdge(
                subject_cui=rel.subject_cui,
                normalized_predicate=pred,
                object_cui=rel.object_cui,
                polarity=rel.polarity, modality=rel.modality,
                conditions=rel.conditions,
                negated=negated, evidence_count=0,
                supporting_relation_ids=[])
        agg[key].evidence_count += 1
        agg[key].supporting_relation_ids.append(rel.relation_id)

    return list(agg.values())


# ---------------------------------------------------------------------
#  Agent setup (call once per run)
# ---------------------------------------------------------------------

def register_agents(recorder: Recorder) -> None:
    recorder.agent("agent_extraction", cls=KGP.ExtractionAgent)
    recorder.agent("agent_evaluator", cls=KGP.PrimaryEvaluatorAgent)
    recorder.agent("agent_escalation", cls=KGP.EscalationSearchAgent,
                   on_behalf_of="agent_evaluator")
    recorder.agent("agent_relation", cls=KGP.RelationExtractionAgent)
    recorder.agent("agent_normalization", cls=KGP.NormalizationAgent)
    recorder.agent("agent_retrieval", cls=KGP.PrimaryEvaluatorAgent)
