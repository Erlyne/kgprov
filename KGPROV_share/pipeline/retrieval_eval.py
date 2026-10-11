"""Reachability evaluation over BC5CDR gold mentions.

For every gold mention with a MeSH ID, run the six-stage retrieval pipeline
and record where the gold UI appears at each stage. Output columns are
listed in ``_COLUMNS``.

What this measures
------------------
The *reachability* half of this module is exact: whether the gold
descriptor survived into the final constraint space is a property of the
space, decided entirely by retrieval, and no model is involved.

The *selection* half is a *proxy*. ``selected_id`` is the cross-encoder's
rank-1 candidate, not a language model's choice. A mention is therefore
classified ``REACHABLE_RERANK1_MISS`` whenever gold is in the pool at any
rank other than 1 -- which says nothing about whether Tier 1 would have
picked it. The outcome names carry ``RERANK1`` so that this is not read
as model behaviour.

For the real model decision, use ``tools/exact_selection.py``, which joins
``tier1_arm_decisions.csv`` to gold.

``canonical_outcome()`` also accepts the labels ``REACHABLE_SELECTED`` /
``REACHABLE_NOT_SELECTED`` and maps them to the RERANK1 names when a CSV
is read back.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

from kgprov import KGP, ids
from .loaders import (
    Document, MeshOntology, OntologyEntry, load_bc5cdr, parse_mesh_xml,
)
from .retrieval import MeshIndex, RetrievalParams, RetrievalResult, retrieve


# ---------------------------------------------------------------------
#  Outcome classification
# ---------------------------------------------------------------------

#: Canonical outcome labels. RERANK1 marks the proxy selector so that the
#: retrieval measurement can never be read as a model measurement.
REACHABLE_RERANK1_HIT = "REACHABLE_RERANK1_HIT"
REACHABLE_RERANK1_MISS = "REACHABLE_RERANK1_MISS"
UNREACHABLE = "UNREACHABLE"
GOLD_NOT_IN_ONTOLOGY = "GOLD_NOT_IN_ONTOLOGY"
NO_GOLD = "NO_GOLD"

#: Alternative labels that canonical_outcome() maps to the RERANK1 names.
_LEGACY_ALIASES = {
    "REACHABLE_SELECTED": REACHABLE_RERANK1_HIT,
    "REACHABLE_NOT_SELECTED": REACHABLE_RERANK1_MISS,
}

#: The selector this module applies, recorded in every summary it writes.
SELECTOR = "reranker_top1_proxy"


def canonical_outcome(outcome: str) -> str:
    """Map a legacy outcome label onto its current name."""
    return _LEGACY_ALIASES.get(outcome, outcome)


def classify_outcome(gold_mesh_id: str | None,
                     gold_in_ontology: bool,
                     final_uis: list[str],
                     selected_id: str | None) -> str:
    if gold_mesh_id is None:
        return NO_GOLD
    if not gold_in_ontology:
        return GOLD_NOT_IN_ONTOLOGY
    if gold_mesh_id not in final_uis:
        return UNREACHABLE
    # NB: selected_id is the re-ranker's rank-1, not a model decision.
    if selected_id == gold_mesh_id:
        return REACHABLE_RERANK1_HIT
    return REACHABLE_RERANK1_MISS


# ---------------------------------------------------------------------
#  Per-mention row builder
# ---------------------------------------------------------------------

_COLUMNS = [
    "entity_id", "doc_id", "mention_text", "semantic_category", "gold_mesh_id",
    "gold_in_dense_top200", "gold_in_sparse_top200", "gold_in_fused_top100",
    "gold_survived_type_filter", "gold_in_final_topk",
    "gold_rank_in_final", "gold_score_in_final",
    "space_suppressed_by_tau",
    "selected_id", "selected_correct", "outcome",
]


def _row_from_result(doc: Document, mention_idx: int, ontology: MeshOntology,
                     result: RetrievalResult) -> dict:
    m = doc.mentions[mention_idx]
    eid = ids.entity_id(doc.pmid, 0, m.start, m.end, m.text)
    gold = m.mesh_id
    gold_in_ontology = gold is not None and ontology.contains(gold)

    dense_set = set(result.dense_uis)
    sparse_set = set(result.sparse_uis)
    fused_set = set(result.fused_uis)
    typed_set = set(result.typed_uis)
    final_uis = [ui for ui, _ in result.final]
    final_set = set(final_uis)

    gold_rank = None
    gold_score = None
    if gold is not None:
        for i, (ui, sc) in enumerate(result.final, start=1):
            if ui == gold:
                gold_rank = i
                gold_score = sc
                break

    selected = final_uis[0] if final_uis else None
    outcome = classify_outcome(gold, gold_in_ontology, final_uis, selected)

    return {
        "entity_id": eid,
        "doc_id": doc.pmid,
        "mention_text": m.text,
        "semantic_category": m.entity_type,
        "gold_mesh_id": gold if gold is not None else "",
        "gold_in_dense_top200": gold in dense_set if gold else False,
        "gold_in_sparse_top200": gold in sparse_set if gold else False,
        "gold_in_fused_top100": gold in fused_set if gold else False,
        "gold_survived_type_filter": gold in typed_set if gold else False,
        "gold_in_final_topk": gold in final_set if gold else False,
        "gold_rank_in_final": gold_rank if gold_rank is not None else "",
        "gold_score_in_final": f"{gold_score:.6f}" if gold_score is not None else "",
        "space_suppressed_by_tau": result.suppressed_by_tau,
        "selected_id": selected if selected is not None else "",
        "selected_correct": (gold is not None and selected == gold),
        "outcome": outcome,
    }


# ---------------------------------------------------------------------
#  Aggregation report
# ---------------------------------------------------------------------

def _truthy(v) -> bool:
    """Rows are bools in-process and strings when read back from CSV."""
    return v is True or v == "True"


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    by_outcome: dict[str, int] = {}
    for r in rows:
        o = canonical_outcome(r["outcome"])
        by_outcome[o] = by_outcome.get(o, 0) + 1

    with_gold_in_onto = [r for r in rows
                         if canonical_outcome(r["outcome"])
                         not in (NO_GOLD, GOLD_NOT_IN_ONTOLOGY)]
    n_denom = len(with_gold_in_onto)
    n_errors = sum(1 for r in with_gold_in_onto
                   if canonical_outcome(r["outcome"]) != REACHABLE_RERANK1_HIT)
    n_reach_fail = sum(1 for r in with_gold_in_onto
                       if canonical_outcome(r["outcome"]) == UNREACHABLE)
    n_rerank1_miss = sum(
        1 for r in with_gold_in_onto
        if canonical_outcome(r["outcome"]) == REACHABLE_RERANK1_MISS)

    # CQ3: at which narrowing stage did the gold descriptor leave?
    #
    # The chain is ordered, and the first stage that no longer contains
    # gold is the one that lost it. Gold surfaced by either index has
    # been retrieved, so the walk continues to fusion; only gold missed
    # by both is a retrieval-recall failure. The assertion below checks
    # that the breakdown sums to the UNREACHABLE total.
    stage_loss = {
        "never_retrieved_top200": 0,
        "lost_at_fusion_top100": 0,
        "removed_by_type_filter": 0,
        "suppressed_by_score_floor": 0,
        "lost_at_rerank": 0,
    }
    for r in with_gold_in_onto:
        if canonical_outcome(r["outcome"]) != UNREACHABLE:
            continue
        in_dense = _truthy(r["gold_in_dense_top200"])
        in_sparse = _truthy(r["gold_in_sparse_top200"])
        if not in_dense and not in_sparse:
            stage_loss["never_retrieved_top200"] += 1
        elif not _truthy(r["gold_in_fused_top100"]):
            stage_loss["lost_at_fusion_top100"] += 1
        elif not _truthy(r["gold_survived_type_filter"]):
            stage_loss["removed_by_type_filter"] += 1
        elif _truthy(r["space_suppressed_by_tau"]):
            stage_loss["suppressed_by_score_floor"] += 1
        else:
            stage_loss["lost_at_rerank"] += 1

    assert sum(stage_loss.values()) == n_reach_fail, (
        f"stage breakdown {sum(stage_loss.values())} != "
        f"UNREACHABLE {n_reach_fail}; every reachability failure must be "
        f"attributable to exactly one narrowing stage")

    return {
        "selector": SELECTOR,
        "_selector_note": (
            "RERANK1_HIT/MISS describe the cross-encoder's rank-1 "
            "candidate, not a language model's choice. For the real "
            "Tier-1 decision see tools/exact_selection.py."),
        "n_rows": n,
        "by_outcome": by_outcome,
        "n_gold_in_ontology": n_denom,
        "reachability_rate_of_all_gold_in_onto": (
            round((n_denom - n_reach_fail) / n_denom, 4) if n_denom else None),
        "n_errors": n_errors,
        "pct_errors_reachability": (
            round(n_reach_fail / n_errors * 100, 2) if n_errors else None),
        "pct_errors_rerank1_miss": (
            round(n_rerank1_miss / n_errors * 100, 2) if n_errors else None),
        "reachability_failure_stage_loss": stage_loss,
    }


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def _iter_gold_mentions(docs: list[Document]):
    for doc in docs:
        for idx, m in enumerate(doc.mentions):
            yield doc, idx, m


def subsample_ontology(ont: MeshOntology, coverage: float,
                       seed: int) -> MeshOntology:
    """Uniformly-random descriptor keep-set at the given coverage.

    Gold answers are not preferentially kept. Seed is logged so the
    sample is reproducible.
    """
    if coverage >= 1.0:
        return ont
    rng = random.Random(seed)
    all_uis = list(ont.entries.keys())
    rng.shuffle(all_uis)
    n_keep = int(len(all_uis) * coverage)
    keep = set(all_uis[:n_keep])
    new_entries = {ui: e for ui, e in ont.entries.items() if ui in keep}
    new_surface = [(s, ui) for (s, ui) in ont.surface_forms if ui in keep]
    return MeshOntology(
        version=f"{ont.version}_cov{coverage:.2f}_s{seed}",
        entries=new_entries,
        surface_forms=new_surface,
    )


def _gold_by_key(keys: dict) -> dict[tuple, str]:
    """Majority gold descriptor per deduplication key.

    Retrieval runs once per key while gold is annotated per mention, so a
    key covering mentions with different gold answers needs one witness
    value. The majority is taken and the ambiguity is reported by the
    caller rather than hidden.
    """
    from collections import Counter
    out: dict[tuple, str] = {}
    for key, mentions_at_key in keys.items():
        c = Counter(m.mesh_id for _d, _i, m in mentions_at_key if m.mesh_id)
        if c:
            out[key] = c.most_common(1)[0][0]
    return out


def run_eval(docs: list[Document], ontology: MeshOntology,
             index: MeshIndex, params: RetrievalParams,
             out_csv: Path, emit_reranked_pkl: Path | None = None,
             witness_ttl: Path | None = None,
             run_id: str = "witness") -> dict:
    """Evaluate reachability, optionally emitting a gold-witness graph.

    ``witness_ttl`` writes a provenance graph in which every narrowing
    activity carries ``kgp:goldSurvived``, making stage attribution a
    graph query.

    It is written separately from the pipeline's own provenance because
    the pipeline must never see gold. A gold annotation is an assessment
    of a derivation, not part of one, so it is attached via ``assesses``
    rather than ``used``; a separate graph keeps that boundary structural.
    """
    # Dedup: unique (norm(text), category) -> RetrievalResult
    print("[eval] deduplicating queries")
    keys: dict[tuple[str, str], list[tuple[Document, int]]] = {}
    keys_full: dict[tuple[str, str], list] = {}
    for doc, idx, m in _iter_gold_mentions(docs):
        key = (ids.norm(m.text), m.entity_type)
        keys.setdefault(key, []).append((doc, idx))
        keys_full.setdefault(key, []).append((doc, idx, m))
    n_unique = len(keys)
    print(f"[eval] {sum(len(v) for v in keys.values())} mentions, "
          f"{n_unique} unique (norm-text, category) keys")

    gold_by_key = _gold_by_key(keys_full) if witness_ttl else {}
    ambiguous = sum(
        1 for k, v in keys_full.items()
        if len({m.mesh_id for _d, _i, m in v if m.mesh_id}) > 1)
    if witness_ttl:
        print(f"[eval] gold witness enabled; {len(gold_by_key)} keys with "
              f"gold, {ambiguous} keys whose mentions disagree on gold "
              f"(majority taken)")

    recorder = None
    if witness_ttl:
        from kgprov import Recorder
        recorder = Recorder(run_id, config_variant=f"{run_id}_witness",
                            model_id="none",
                            ontology_version=ontology.version)
        recorder.agent("agent_retrieval")

    print(f"[eval] running retrieval over {n_unique} unique queries")
    results: dict[tuple[str, str], RetrievalResult] = {}
    t0 = time.time()
    for i, (key, mentions_at_key) in enumerate(keys.items(), start=1):
        norm_text, category = key
        # Use the first mention's original text as the query (readability).
        first_doc, first_idx = mentions_at_key[0]
        query = first_doc.mentions[first_idx].text
        try:
            r = retrieve(index, query, category, params,
                         recorder=recorder,
                         discriminators=(norm_text, category),
                         gold=gold_by_key.get(key))
        except Exception as e:
            print(f"[eval] ERROR on '{query}' ({category}): {e}")
            r = RetrievalResult(
                query=query, category=category,
                dense_uis=[], sparse_uis=[], fused_uis=[], typed_uis=[],
                reranked=[], final=[], suppressed_by_tau=False)
        results[key] = r

        if i % 100 == 0 or i == n_unique:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n_unique - i) / rate if rate > 0 else 0
            print(f"[eval] {i}/{n_unique} ({rate:.1f} q/s, ETA {eta:.0f}s)")

    print(f"[eval] writing {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for key, mentions_at_key in keys.items():
            r = results[key]
            for doc, idx in mentions_at_key:
                row = _row_from_result(doc, idx, ontology, r)
                w.writerow(row)
                rows.append(row)

    if emit_reranked_pkl is not None:
        # Persist enough per-(dedup-key) info to post-process for other
        # τ / k values without re-running the cross-encoder.
        emit = {}
        for key, r in results.items():
            emit[key] = {
                "reranked": r.reranked,  # full (ui, score) list, sorted desc
                "typed_uis": r.typed_uis,
                "fused_uis": r.fused_uis,
                "dense_uis": r.dense_uis,
                "sparse_uis": r.sparse_uis,
            }
        emit_reranked_pkl.parent.mkdir(parents=True, exist_ok=True)
        with emit_reranked_pkl.open("wb") as f:
            pickle.dump(emit, f)
        print(f"[eval] wrote reranked pkl to {emit_reranked_pkl}")

    if recorder is not None and witness_ttl is not None:
        # Attach each gold value as an assessment, never as an input: the
        # retrieval never consumed it, and asserting that it did would
        # make a lineage claim that does not hold.
        for key, gold in gold_by_key.items():
            first_doc, first_idx = keys[key][0]
            m = first_doc.mentions[first_idx]
            eid = ids.entity_id(first_doc.pmid, 0, m.start, m.end, m.text)
            recorder.entity(eid, cls=KGP.GroundedMention, rawSpan=m.text)
            recorder.gold_annotation(f"gold_{eid}", assesses=eid, value=gold)
        witness_ttl.parent.mkdir(parents=True, exist_ok=True)
        recorder.serialize(str(witness_ttl))
        print(f"[eval] wrote gold-witness graph to {witness_ttl} "
              f"({len(recorder)} triples)")

    summary = _summarize(rows)
    summary["gold_witness_graph"] = str(witness_ttl) if witness_ttl else None
    summary["n_keys_with_ambiguous_gold"] = ambiguous
    return summary


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    ap.add_argument("--n-docs", type=int, default=None,
                    help="Cap number of documents (smoke test)")
    ap.add_argument("--bc5cdr-dir", default="data/bc5cdr")
    ap.add_argument("--mesh-xml", default="data/mesh/desc2025.xml")
    ap.add_argument("--mesh-version", default="2025")
    ap.add_argument("--index-cache", default="data/mesh/index_2025")
    ap.add_argument("--out-csv", default=None,
                    help="Default: out/results/reachability_<split>.csv")
    ap.add_argument("--dense-top-k", type=int, default=200)
    ap.add_argument("--sparse-top-k", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--fused-top-k", type=int, default=100)
    ap.add_argument("--rerank-top-k", type=int, default=5)
    ap.add_argument("--score-floor", type=float, default=0.15)
    ap.add_argument("--no-score-floor", action="store_true",
                    help="Disable score floor entirely; equivalent to "
                    "τ=-inf. Useful for post-hoc τ ablation.")
    ap.add_argument("--coverage", type=float, default=1.0,
                    help="Fraction of descriptors to keep (seeded random).")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--emit-reranked", action="store_true",
                    help="Also write a pickle of per-key full reranked "
                    "lists so τ/k variants can be computed post-hoc.")
    ap.add_argument("--emit-witness", action="store_true",
                    help="Also write a provenance graph carrying "
                         "kgp:goldSurvived per narrowing stage, so stage "
                         "attribution is a graph query rather than a "
                         "spreadsheet. Written separately from the "
                         "pipeline's own graph: the pipeline must never "
                         "see gold.")
    args = ap.parse_args()

    print(f"[eval] loading BC5CDR split={args.split}")
    docs = load_bc5cdr(args.bc5cdr_dir, args.split)
    if args.n_docs:
        docs = docs[:args.n_docs]
        print(f"[eval] capped to {len(docs)} docs")

    print(f"[eval] loading MeSH {args.mesh_version} from {args.mesh_xml}")
    ontology = parse_mesh_xml(args.mesh_xml, args.mesh_version)
    print(f"[eval] {len(ontology.entries)} descriptors, "
          f"{len(ontology.surface_forms)} surface forms")

    index_cache = args.index_cache
    if args.coverage < 1.0:
        ontology = subsample_ontology(ontology, args.coverage, args.seed)
        print(f"[eval] subsampled coverage={args.coverage} seed={args.seed} "
              f"-> {len(ontology.entries)} descriptors, "
              f"{len(ontology.surface_forms)} surface forms")
        index_cache = f"{args.index_cache}_cov{args.coverage:.2f}_s{args.seed}"

    index = MeshIndex.build_or_load(ontology, index_cache)

    params = RetrievalParams(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rrf_k=args.rrf_k,
        fused_top_k=args.fused_top_k,
        rerank_top_k=args.rerank_top_k,
        score_floor=(-1e9 if args.no_score_floor else args.score_floor),
    )

    out_csv = Path(args.out_csv) if args.out_csv else \
        Path("out/results") / f"reachability_{args.split}.csv"

    pkl_path = None
    if args.emit_reranked:
        pkl_path = out_csv.with_suffix(".reranked.pkl")

    witness_path = None
    if args.emit_witness:
        witness_path = (Path("out/provenance") /
                        f"{out_csv.stem}_witness.ttl")

    summary = run_eval(docs, ontology, index, params, out_csv,
                       emit_reranked_pkl=pkl_path,
                       witness_ttl=witness_path,
                       run_id=out_csv.stem)

    print("\n=== SUMMARY ===")
    import json
    print(json.dumps(summary, indent=2))

    # Also persist the summary next to the CSV
    (out_csv.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2))
    print(f"\n[eval] wrote {out_csv}")
    print(f"[eval] wrote {out_csv.with_suffix('.summary.json')}")


if __name__ == "__main__":
    _main()
