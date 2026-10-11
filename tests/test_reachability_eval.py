"""Outcome classification and stage attribution.

`classify_outcome` produces the reachability/selection decomposition.
The stage-attribution tests check that every reachability failure is
attributed to a stage, so the per-stage breakdown sums to the
UNREACHABLE total.
"""

from __future__ import annotations

import pytest

from pipeline.retrieval_eval import (
    GOLD_NOT_IN_ONTOLOGY, NO_GOLD, REACHABLE_RERANK1_HIT,
    REACHABLE_RERANK1_MISS, UNREACHABLE, _summarize, canonical_outcome,
    classify_outcome,
)


# ---------------------------------------------------------------------
#  classify_outcome
# ---------------------------------------------------------------------

def test_no_gold_takes_precedence():
    assert classify_outcome(None, False, ["D1"], "D1") == NO_GOLD


def test_gold_outside_ontology_is_its_own_class():
    """Gold absent from the ontology is not a reachability failure: no
    retrieval stack can surface a descriptor the ontology lacks."""
    assert classify_outcome("D9", False, ["D1"], "D1") == GOLD_NOT_IN_ONTOLOGY


def test_gold_absent_from_final_pool_is_unreachable():
    assert classify_outcome("D9", True, ["D1", "D2"], "D1") == UNREACHABLE


def test_gold_at_rank_one_is_a_rerank1_hit():
    assert classify_outcome("D1", True, ["D1", "D2"], "D1") == REACHABLE_RERANK1_HIT


def test_gold_present_but_not_rank_one_is_a_rerank1_miss():
    assert classify_outcome("D2", True, ["D1", "D2"], "D1") == REACHABLE_RERANK1_MISS


def test_empty_pool_is_unreachable():
    assert classify_outcome("D1", True, [], None) == UNREACHABLE


def test_outcome_names_mark_the_proxy():
    """The label must not imply a model made this decision.

    `selected_id` is the cross-encoder's rank-1 candidate, so the outcome
    names say RERANK1 rather than SELECTED.
    """
    assert "RERANK1" in REACHABLE_RERANK1_HIT
    assert "RERANK1" in REACHABLE_RERANK1_MISS
    assert "SELECTED" not in REACHABLE_RERANK1_HIT


def test_legacy_labels_still_map_forward():
    assert canonical_outcome("REACHABLE_SELECTED") == REACHABLE_RERANK1_HIT
    assert canonical_outcome("REACHABLE_NOT_SELECTED") == REACHABLE_RERANK1_MISS
    assert canonical_outcome(UNREACHABLE) == UNREACHABLE


# ---------------------------------------------------------------------
#  Stage attribution
# ---------------------------------------------------------------------

def _row(**kw):
    base = dict(
        outcome=UNREACHABLE,
        gold_in_dense_top200=False, gold_in_sparse_top200=False,
        gold_in_fused_top100=False, gold_survived_type_filter=False,
        space_suppressed_by_tau=False,
    )
    base.update(kw)
    return base


def test_gold_missed_by_both_indices_is_a_retrieval_loss():
    s = _summarize([_row()])
    assert s["reachability_failure_stage_loss"]["never_retrieved_top200"] == 1


def test_gold_found_by_sparse_only_continues_down_the_chain():
    """Gold found by BM25 but missed by the dense index has been retrieved.

    The walk continues down the chain and attributes the loss to the
    stage that removed it, here the score floor.
    """
    s = _summarize([_row(gold_in_sparse_top200=True,
                         gold_in_fused_top100=True,
                         gold_survived_type_filter=True,
                         space_suppressed_by_tau=True)])
    loss = s["reachability_failure_stage_loss"]
    assert loss["suppressed_by_score_floor"] == 1
    assert loss["never_retrieved_top200"] == 0


def test_stage_breakdown_sums_to_unreachable():
    rows = [
        _row(),
        _row(gold_in_dense_top200=True),
        _row(gold_in_dense_top200=True, gold_in_fused_top100=True),
        _row(gold_in_dense_top200=True, gold_in_fused_top100=True,
             gold_survived_type_filter=True, space_suppressed_by_tau=True),
        _row(gold_in_dense_top200=True, gold_in_fused_top100=True,
             gold_survived_type_filter=True),
        dict(outcome=REACHABLE_RERANK1_HIT, gold_in_dense_top200=True,
             gold_in_sparse_top200=True, gold_in_fused_top100=True,
             gold_survived_type_filter=True, space_suppressed_by_tau=False),
    ]
    s = _summarize(rows)
    assert (sum(s["reachability_failure_stage_loss"].values())
            == s["by_outcome"][UNREACHABLE] == 5)


def test_summary_records_which_selector_produced_it():
    s = _summarize([_row()])
    assert s["selector"] == "reranker_top1_proxy"


# ---------------------------------------------------------------------
#  Against the real ledger, when one is present
# ---------------------------------------------------------------------

def test_real_ledger_stage_breakdown_sums(reachability_rows):
    s = _summarize(reachability_rows)
    assert (sum(s["reachability_failure_stage_loss"].values())
            == s["by_outcome"][UNREACHABLE])


def test_real_ledger_rerank1_miss_never_has_gold_at_rank_one(reachability_rows):
    """Checks the selector is a deterministic argmax, not a model.

    If anything other than rank 1 were choosing, some RERANK1_MISS row
    would have gold at rank 1.
    """
    ranks = {r["gold_rank_in_final"] for r in reachability_rows
             if canonical_outcome(r["outcome"]) == REACHABLE_RERANK1_MISS}
    assert "1" not in ranks
