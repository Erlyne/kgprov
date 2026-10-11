"""Retrieval stack: SapBERT + FAISS + BM25 + six-stage narrowing chain.

Stages, by the kgp:narrowingStage label each records:
  1 DENSE_RETRIEVAL      -> top 200 unique UIs by best surface-form rank
  2 SPARSE_RETRIEVAL     -> top 200 unique UIs by best surface-form rank
  3 RANK_FUSION          -> RRF (k=60), dedup, truncate to top 100
  4 TYPE_FILTER          -> drop UIs whose types don't include mention category
  5 CROSS_ENCODER_RERANK -> re-rank survivors, keep top rerank_top_k
  6 SCORE_FLOOR          -> suppress the whole set if best < tau

Every stage returns its intermediate result so retrieval_eval.py can fill
the columns of reachability_<split>.csv. The Recorder is optional;
when passed, each stage wraps in `recorder.narrowing()`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from rdflib import Literal
from rdflib.namespace import XSD

from kgprov.capture import (
    KGP, ConstraintSpace, Recorder, SpaceMember,
)

from .loaders import MeshOntology


# ---------------------------------------------------------------------
#  Result container
# ---------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Everything needed to fill one query's reachability_<split>.csv columns.

    Every stage exposes the UI set that survived it (or the ranked scores
    for the final two stages), so the eval can check whether the gold UI
    was present at each transition.
    """
    query: str
    category: str
    dense_uis: list[str]                    # up to top_k unique UIs
    sparse_uis: list[str]
    fused_uis: list[str]                    # ≤ 100
    typed_uis: list[str]
    reranked: list[tuple[str, float]]       # (ui, score), full list before top-k cut
    final: list[tuple[str, float]]          # ≤ rerank_top_k; empty if suppressed
    suppressed_by_tau: bool
    space_id: str | None = None             # id of the final ConstraintSpace


# ---------------------------------------------------------------------
#  Index
# ---------------------------------------------------------------------

_ENCODER_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _bm25_tokenize(s: str) -> list[str]:
    """Lowercase, split on whitespace and hyphens."""
    out: list[str] = []
    for tok in s.lower().split():
        for sub in tok.split("-"):
            sub = sub.strip()
            if sub:
                out.append(sub)
    return out


class MeshIndex:
    """FAISS + BM25 over one MeSH release's BC5CDR-compatible surface forms.

    Cached to disk so it is built once per version.
    """

    def __init__(self, version: str,
                 surface_forms: list[tuple[str, str]],
                 types_by_ui: dict[str, set[str]],
                 faiss_index,   # faiss.IndexFlatIP
                 encoder,       # sentence_transformers.SentenceTransformer or None
                 reranker=None):
        self.version = version
        self.surface_forms = surface_forms         # parallel to faiss rows
        self.types_by_ui = types_by_ui
        self._faiss = faiss_index
        self._encoder = encoder
        self._reranker = reranker
        # Rebuild BM25 in memory (fast at this scale).
        from rank_bm25 import BM25Okapi
        tokenized = [_bm25_tokenize(s) for s, _ in surface_forms]
        self._bm25 = BM25Okapi(tokenized)

    # -- construction -------------------------------------------------

    @classmethod
    def build(cls, ontology: MeshOntology, cache_dir: str | Path,
              batch_size: int = 64) -> "MeshIndex":
        """Build a new index from an ontology. Slow (embeds ~150k strings)."""
        import faiss  # local import so import-time cost is only paid on use
        from sentence_transformers import SentenceTransformer

        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        surface_forms = ontology.surface_forms
        types_by_ui = {ui: set(e.entity_types)
                       for ui, e in ontology.entries.items()}

        print(f"[index] loading encoder {_ENCODER_MODEL}")
        encoder = SentenceTransformer(_ENCODER_MODEL)

        print(f"[index] encoding {len(surface_forms)} surface forms "
              f"(batch_size={batch_size})")
        texts = [s for s, _ in surface_forms]
        emb = encoder.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            show_progress_bar=True, normalize_embeddings=True)
        emb = emb.astype("float32")

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        # Persist.
        faiss.write_index(index, str(cache / "faiss.index"))
        with (cache / "surface_forms.pkl").open("wb") as f:
            pickle.dump({"surface_forms": surface_forms,
                         "types_by_ui": types_by_ui,
                         "version": ontology.version}, f)
        print(f"[index] cached to {cache}")

        return cls(version=ontology.version,
                   surface_forms=surface_forms,
                   types_by_ui=types_by_ui,
                   faiss_index=index,
                   encoder=encoder)

    @classmethod
    def load(cls, cache_dir: str | Path,
             load_encoder: bool = True,
             load_reranker: bool = True) -> "MeshIndex":
        import faiss
        cache = Path(cache_dir)
        with (cache / "surface_forms.pkl").open("rb") as f:
            meta = pickle.load(f)
        index = faiss.read_index(str(cache / "faiss.index"))

        encoder = None
        if load_encoder:
            from sentence_transformers import SentenceTransformer
            print(f"[index] loading encoder {_ENCODER_MODEL}")
            encoder = SentenceTransformer(_ENCODER_MODEL)

        reranker = None
        if load_reranker:
            from sentence_transformers import CrossEncoder
            print(f"[index] loading cross-encoder {_RERANKER_MODEL}")
            reranker = CrossEncoder(_RERANKER_MODEL)

        return cls(version=meta["version"],
                   surface_forms=meta["surface_forms"],
                   types_by_ui=meta["types_by_ui"],
                   faiss_index=index,
                   encoder=encoder,
                   reranker=reranker)

    @classmethod
    def build_or_load(cls, ontology: MeshOntology, cache_dir: str | Path,
                      load_reranker: bool = True) -> "MeshIndex":
        cache = Path(cache_dir)
        marker = cache / "faiss.index"
        if marker.exists():
            print(f"[index] loading cached index from {cache}")
            return cls.load(cache, load_encoder=True, load_reranker=load_reranker)
        print(f"[index] no cache at {cache}, building fresh")
        idx = cls.build(ontology, cache)
        if load_reranker:
            from sentence_transformers import CrossEncoder
            print(f"[index] loading cross-encoder {_RERANKER_MODEL}")
            idx._reranker = CrossEncoder(_RERANKER_MODEL)
        return idx

    # -- lookups ------------------------------------------------------

    def preferred_name(self, ui: str) -> str | None:
        # First surface form found for this UI is the preferred name
        # (parse_mesh_xml emits the preferred name first).
        for s, u in self.surface_forms:
            if u == ui:
                return s
        return None

    # -- stages -------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        assert self._encoder is not None, "encoder not loaded"
        q = self._encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True)
        return q.astype("float32")

    def dense_search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        """Return top_k unique UIs by best surface-form rank."""
        q = self.encode_query(query)
        # over-fetch so ~top_k unique UIs remain after dedup
        fetch = min(max(top_k * 3, top_k + 50), len(self.surface_forms))
        scores, idxs = self._faiss.search(q, fetch)
        best: dict[str, float] = {}
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            _, ui = self.surface_forms[idx]
            if ui not in best or score > best[ui]:
                best[ui] = score
            if len(best) >= top_k:
                break
        return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def sparse_search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        tokens = _bm25_tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # top_k * 3 dedup buffer
        fetch = min(max(top_k * 3, top_k + 50), len(scores))
        top_idx = np.argpartition(-scores, fetch - 1)[:fetch]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        best: dict[str, float] = {}
        for idx in top_idx.tolist():
            _, ui = self.surface_forms[idx]
            sc = float(scores[idx])
            if ui not in best or sc > best[ui]:
                best[ui] = sc
            if len(best) >= top_k:
                break
        return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------
#  Rank fusion
# ---------------------------------------------------------------------

def reciprocal_rank_fusion(dense: list[tuple[str, float]],
                           sparse: list[tuple[str, float]],
                           k_rrf: int = 60,
                           top_k: int = 100) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (ui, _) in enumerate(dense, start=1):
        scores[ui] = scores.get(ui, 0.0) + 1.0 / (k_rrf + rank)
    for rank, (ui, _) in enumerate(sparse, start=1):
        scores[ui] = scores.get(ui, 0.0) + 1.0 / (k_rrf + rank)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


# ---------------------------------------------------------------------
#  End-to-end
# ---------------------------------------------------------------------

@dataclass
class RetrievalParams:
    dense_top_k: int = 200
    sparse_top_k: int = 200
    rrf_k: int = 60
    fused_top_k: int = 100
    rerank_top_k: int = 5
    score_floor: float = 0.15


def _ontology_input_space(version: str) -> ConstraintSpace:
    """Symbolic input space representing the whole MeSH index.

    Cardinality shown in the graph is 1 (the symbolic member). Recording
    ~150k members per retrieval call would explode the graph.
    """
    return ConstraintSpace(
        kind="mesh_index",
        members=[SpaceMember(value=f"MESH_{version}",
                             label=f"MeSH {version} surface index",
                             rank=1, selected=False)],
        derived=False,
        discriminators=(version,),
    )


def _cs(kind: str, uis_with_scores: list[tuple[str, float]],
        *, discriminators: tuple, selected: str | None = None) -> ConstraintSpace:
    members = []
    for i, (ui, score) in enumerate(uis_with_scores, start=1):
        members.append(SpaceMember(
            value=ui, rank=i, score=float(score),
            selected=(ui == selected)))
    return ConstraintSpace(kind=kind, members=members, derived=True,
                           discriminators=discriminators)


def _fix_cardinality(recorder: Recorder, activity_uri, true_in: int,
                     true_out: int) -> None:
    """Overwrite input/outputCardinality on a narrowing activity.

    Needed when empty-member ConstraintSpaces are passed to keep the graph
    compact: the recorder's automatic cardinality would then be 0 and
    CQ3's SUM(in - out) would be wrong.
    """
    g = recorder.g
    g.remove((activity_uri, KGP.inputCardinality, None))
    g.add((activity_uri, KGP.inputCardinality,
           Literal(true_in, datatype=XSD.integer)))
    g.remove((activity_uri, KGP.outputCardinality, None))
    g.add((activity_uri, KGP.outputCardinality,
           Literal(true_out, datatype=XSD.integer)))


def _empty_space(kind: str, discriminators: tuple) -> ConstraintSpace:
    return ConstraintSpace(kind=kind, members=[], derived=True,
                           discriminators=discriminators)


def retrieve(index: MeshIndex, query: str, category: str,
             params: RetrievalParams,
             *, recorder: Recorder | None = None,
             discriminators: tuple = (),
             record_full_scores: bool = True,
             compact_narrowing: bool = True,
             gold: str | None = None) -> RetrievalResult:
    """Run the six-stage narrowing chain for one (query, category).

    Args:
        index: MeshIndex.
        query: the resolved_string to look up.
        category: "Chemical" or "Disease".
        params: retrieval parameters.
        recorder: optional Recorder; when None, no provenance is emitted
            but the RetrievalResult is fully populated.
        discriminators: extra components mixed into activity/space IDs
            (typically (doc_id, entity_id) or (dedup_key,)).
        record_full_scores: used only when compact_narrowing is False.
            If True, the reranker's output space lists every
            post-type-filter survivor with its cross-encoder score; if
            False, only the kept top-k.
    """
    disc = discriminators
    input_space = _ontology_input_space(index.version)

    # In compact mode, use empty-member ConstraintSpaces for intermediate
    # stages and override cardinality afterwards. Keeps CQ3 populated
    # (needs inputCardinality/outputCardinality/exclusionReason) at a
    # small fraction of the triple cost.

    # -- Step 1: Dense ---------------------------------------------------
    dense = index.dense_search(query, params.dense_top_k)
    dense_uis = [ui for ui, _ in dense]
    if recorder is not None:
        in_sp = (input_space if not compact_narrowing
                 else _empty_space("mesh_index_compact", (index.version,)))
        with recorder.narrowing(
            "DENSE_RETRIEVAL", input_space=in_sp, gold=gold,
            agent="agent_retrieval", discriminators=("dense",) + disc) as n:
            out_sp = (_empty_space("dense_top_k", ("dense",) + disc)
                      if compact_narrowing
                      else _cs("dense_top_k", dense,
                               discriminators=("dense",) + disc))
            n.output_space(out_sp, true_members=dense_uis)
            if compact_narrowing:
                _fix_cardinality(recorder, n.uri,
                                 len(index.surface_forms), len(dense))

    # -- Step 2: Sparse --------------------------------------------------
    sparse = index.sparse_search(query, params.sparse_top_k)
    sparse_uis = [ui for ui, _ in sparse]
    if recorder is not None:
        in_sp = (input_space if not compact_narrowing
                 else _empty_space("mesh_index_compact", (index.version,)))
        with recorder.narrowing(
            "SPARSE_RETRIEVAL", input_space=in_sp, gold=gold,
            agent="agent_retrieval", discriminators=("sparse",) + disc) as n:
            out_sp = (_empty_space("sparse_top_k", ("sparse",) + disc)
                      if compact_narrowing
                      else _cs("sparse_top_k", sparse,
                               discriminators=("sparse",) + disc))
            n.output_space(out_sp, true_members=sparse_uis)
            if compact_narrowing:
                _fix_cardinality(recorder, n.uri,
                                 len(index.surface_forms), len(sparse))

    # -- Step 3: Rank fusion --------------------------------------------
    fused = reciprocal_rank_fusion(
        dense, sparse, k_rrf=params.rrf_k, top_k=params.fused_top_k)
    fused_uis = [ui for ui, _ in fused]

    if recorder is not None:
        # Compute the union just for cardinality; skip enumeration in compact.
        union_size = len({ui for ui, _ in dense + sparse})
        if compact_narrowing:
            fusion_input = _empty_space("fusion_input", ("fusion_in",) + disc)
        else:
            union: dict[str, float] = {}
            for ui, sc in dense + sparse:
                if ui not in union or sc > union[ui]:
                    union[ui] = sc
            union_list = sorted(union.items(), key=lambda x: x[1], reverse=True)
            fusion_input = _cs("fusion_input", union_list,
                               discriminators=("fusion_in",) + disc)

        with recorder.narrowing(
            "RANK_FUSION", input_space=fusion_input, gold=gold,
            agent="agent_retrieval", discriminators=("fusion",) + disc) as n:
            if compact_narrowing:
                n.excluded([], reason="BELOW_FUSION_TOP_K")
                n.output_space(_empty_space("fused_top_k",
                                            ("fusion",) + disc),
                               true_members=fused_uis)
                _fix_cardinality(recorder, n.uri, union_size, len(fused))
            else:
                excluded_fusion = [ui for ui in union.keys()
                                   if ui not in set(fused_uis)]
                n.excluded(excluded_fusion, reason="BELOW_FUSION_TOP_K")
                n.output_space(_cs("fused_top_k", fused,
                                   discriminators=("fusion",) + disc))

    # -- Step 4: Type filter --------------------------------------------
    typed: list[tuple[str, float]] = []
    excluded_type: list[str] = []
    for ui, sc in fused:
        types = index.types_by_ui.get(ui, set())
        if category in types:
            typed.append((ui, sc))
        else:
            excluded_type.append(ui)
    typed_uis = [ui for ui, _ in typed]

    if recorder is not None:
        in_sp = (_empty_space("fused_top_k", ("fusion",) + disc)
                 if compact_narrowing
                 else _cs("fused_top_k", fused,
                          discriminators=("fusion",) + disc))
        with recorder.narrowing(
            "TYPE_FILTER", input_space=in_sp, gold=gold,
            agent="agent_retrieval", discriminators=("type",) + disc) as n:
            if compact_narrowing:
                n.excluded([], reason="TYPE_INCOMPATIBLE")
                # Placeholder so that a query counting activities with
                # kgp:excluded can still detect that this stage removed
                # candidates, and the compact graph answers such a query
                # the same way as the full one.
                if excluded_type:
                    from kgprov.capture import iri as _iri
                    recorder.g.add(
                        (n.uri, KGP.excluded,
                         _iri("compact_excluded_placeholder")))
                n.output_space(_empty_space("typed", ("type",) + disc),
                               true_members=typed_uis)
                _fix_cardinality(recorder, n.uri, len(fused), len(typed))
            else:
                n.excluded(excluded_type, reason="TYPE_INCOMPATIBLE")
                n.output_space(_cs("typed", typed,
                                   discriminators=("type",) + disc))

    # -- Step 5: Cross-encoder rerank -----------------------------------
    if not typed:
        reranked: list[tuple[str, float]] = []
    else:
        assert index._reranker is not None, "reranker not loaded"
        names: list[str] = []
        for ui, _ in typed:
            name = index.preferred_name(ui) or ui
            names.append(name)
        pairs = [(query, name) for name in names]
        scores = index._reranker.predict(pairs)
        # cross-encoder scores are unbounded floats (logits); leave as-is
        reranked = sorted(
            [(ui, float(scores[i])) for i, (ui, _) in enumerate(typed)],
            key=lambda x: x[1], reverse=True)

    top_k_after_rerank = reranked[: params.rerank_top_k]

    if recorder is not None and reranked:
        below_top_k = [ui for ui, _ in reranked[params.rerank_top_k:]]
        in_sp = (_empty_space("typed", ("type",) + disc)
                 if compact_narrowing
                 else _cs("typed", typed,
                          discriminators=("type",) + disc))
        with recorder.narrowing(
            "CROSS_ENCODER_RERANK", input_space=in_sp, gold=gold,
            agent="agent_retrieval", discriminators=("rerank",) + disc) as n:
            if compact_narrowing:
                n.excluded([], reason="BELOW_TOP_K")
                # One placeholder excluded triple so exclusion-counting
                # queries can detect that this stage did drop candidates.
                if below_top_k:
                    from kgprov.capture import iri as _iri
                    recorder.g.add(
                        (n.uri, KGP.excluded, _iri("compact_excluded_placeholder")))
                # Known limitation: in compact mode the reranker's output
                # space is recorded with this stage's input cardinality
                # (the whole scored pool) rather than the top-k it kept, so
                # goldSurvived here means "was scored". The activity's own
                # cardinalities are correct.
                n.output_space(_empty_space("reranked", ("rerank",) + disc),
                               true_members=[u for u, _ in reranked])
                _fix_cardinality(recorder, n.uri, len(typed),
                                 min(params.rerank_top_k, len(reranked)))
            else:
                rerank_output = (reranked if record_full_scores
                                 else top_k_after_rerank)
                n.excluded(below_top_k, reason="BELOW_TOP_K")
                n.output_space(_cs("reranked", rerank_output,
                                   discriminators=("rerank",) + disc))

    # -- Step 6: Score floor --------------------------------------------
    suppressed = False
    final = top_k_after_rerank
    if final:
        best_score = final[0][1]
        if best_score < params.score_floor:
            suppressed = True
            if recorder is not None:
                in_sp = (_empty_space("reranked", ("rerank",) + disc)
                         if compact_narrowing
                         else _cs("reranked", top_k_after_rerank,
                                  discriminators=("rerank",) + disc))
                with recorder.narrowing(
                    "SCORE_FLOOR", input_space=in_sp, gold=gold,
                    agent="agent_retrieval",
                    discriminators=("floor",) + disc) as n:
                    if compact_narrowing:
                        n.excluded([], reason="BELOW_SCORE_FLOOR")
                        from kgprov.capture import iri as _iri
                        recorder.g.add(
                            (n.uri, KGP.excluded,
                             _iri("compact_excluded_placeholder")))
                        n.output_space(_empty_space("final",
                                                    ("floor",) + disc),
                                       true_members=[])
                        _fix_cardinality(recorder, n.uri,
                                         len(top_k_after_rerank), 0)
                    else:
                        n.excluded([ui for ui, _ in top_k_after_rerank],
                                   reason="BELOW_SCORE_FLOOR")
                        n.output_space(_cs("final", [],
                                           discriminators=("floor",) + disc))
            final = []
        else:
            if recorder is not None:
                in_sp = (_empty_space("reranked", ("rerank",) + disc)
                         if compact_narrowing
                         else _cs("reranked", top_k_after_rerank,
                                  discriminators=("rerank",) + disc))
                with recorder.narrowing(
                    "SCORE_FLOOR", input_space=in_sp, gold=gold,
                    agent="agent_retrieval",
                    discriminators=("floor",) + disc) as n:
                    if compact_narrowing:
                        n.output_space(
                            _empty_space("final", ("floor",) + disc),
                            true_members=[u for u, _ in top_k_after_rerank])
                        _fix_cardinality(recorder, n.uri,
                                         len(top_k_after_rerank),
                                         len(top_k_after_rerank))
                    else:
                        n.output_space(_cs("final", top_k_after_rerank,
                                           discriminators=("floor",) + disc))

    space_id = None
    if recorder is not None:
        # The final space is the same as the last output; capture its id.
        final_space = _cs("final", final, discriminators=("floor",) + disc)
        space_id = final_space.space_id

    return RetrievalResult(
        query=query, category=category,
        dense_uis=dense_uis,
        sparse_uis=sparse_uis,
        fused_uis=fused_uis,
        typed_uis=typed_uis,
        reranked=reranked,
        final=final,
        suppressed_by_tau=suppressed,
        space_id=space_id,
    )
