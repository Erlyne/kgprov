#!/usr/bin/env python3
"""The reachability/selection split, computed over the exact join.

Gold annotations are joined to pipeline decisions through recorded
identifiers, not through normalised mention text. Phase 1 rewrites
strings (dropping stereochemistry prefixes, expanding abbreviations), so
a text match misses mentions the pipeline did handle, and matches a gold
mention to any dedup key in the corpus with the same normalised text.

This tool uses the join the pipeline records:

    gold --kgp:assesses--> entity_id --covered_entity_ids--> decision

Both halves are written at run time. The first is span overlap, computed
by the runner when it attaches BC5CDR gold to a pipeline mention; the
second is recorded at dispatch. No string is compared at any point.

Reachability needs no separate CSV: a gold answer is reachable exactly
when it appears in the candidate pool the arm was shown, and that pool is
`candidate_ids` on the same row.

Usage:
    python tools/exact_selection.py --run-id stage1_weak
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gold_by_entity(graph_path: Path) -> dict[str, str]:
    """entity_id -> gold descriptor, from the run graph's assessments.

    `kgp:assesses` is written by the runner when a gold span overlaps a
    pipeline mention. Nothing in the pipeline consumes these, so reading
    them here, for evaluation, does not put gold into any derivation.
    """
    from rdflib import Graph
    from kgprov import KGP

    g = Graph()
    g.parse(str(graph_path), format="turtle")
    out: dict[str, str] = {}
    for gold_node, ent in g.subject_objects(KGP.assesses):
        vals = list(g.objects(gold_node, KGP.goldValue))
        if not vals:
            continue
        out[str(ent).rsplit("/", 1)[-1]] = str(vals[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="stage1_weak")
    ap.add_argument("--graph", default=None,
                    help="Run graph (default out/provenance/<run-id>_clean.ttl, "
                         "falling back to <run-id>.ttl).")
    ap.add_argument("--primary-arm", default="weak_arm")
    ap.add_argument("--mesh-xml", default="data/mesh/desc2025.xml",
                    help="The loaded ontology. Gold answers absent from it "
                         "are excluded, not counted as unreachable.")
    ap.add_argument("--mesh-version", default="2025")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = ROOT / "out" / "results" / args.run_id
    if args.graph:
        graph = Path(args.graph)
    else:
        clean = ROOT / "out" / "provenance" / f"{args.run_id}_clean.ttl"
        graph = clean if clean.exists() else (
            ROOT / "out" / "provenance" / f"{args.run_id}.ttl")

    gold = gold_by_entity(graph)
    with (res / "tier1_arm_decisions.csv").open(encoding="utf-8") as f:
        arms = [r for r in csv.DictReader(f) if r["arm"] == args.primary_arm]

    # entity_id -> the decision that covered it. A key recurring across
    # documents produces several rows, each covering that document's
    # mentions; together they partition the mentions, so the last writer
    # per entity is the only row that mentions it.
    by_entity: dict[str, dict] = {}
    for r in arms:
        for e in r["covered_entity_ids"].split("|"):
            if e:
                by_entity[e] = r

    # A gold answer absent from the loaded ontology is reported as its own
    # category and never counted as a reachability failure: no retrieval
    # stack can surface a descriptor the ontology does not contain.
    from pipeline.loaders import parse_mesh_xml
    onto = parse_mesh_xml(args.mesh_xml, args.mesh_version)

    counts: Counter[str] = Counter()
    for ent, gold_id in gold.items():
        if not onto.contains(gold_id):
            counts["GOLD_NOT_IN_ONTOLOGY_excluded"] += 1
            continue
        row = by_entity.get(ent)
        if row is None:
            # The pipeline extracted this mention (it carries a gold
            # assessment) but no Tier 1 row covers it. This count should be
            # zero; a non-zero value indicates a bug.
            counts["NO_DECISION_ROW_bug"] += 1
            continue
        pool = [c for c in row["candidate_ids"].split("|") if c]
        outcome, mapped = row["outcome"], (row["mapped_id"] or "")
        if outcome == "NOT_CALLED":
            counts["UNREACHABLE_pool_suppressed"] += 1
        elif gold_id not in pool:
            counts["UNREACHABLE_gold_not_in_pool"] += 1
        elif outcome == "DECLINED":
            counts["REACHABLE_MODEL_DECLINED"] += 1
        elif mapped == gold_id:
            counts["REACHABLE_MODEL_CORRECT"] += 1
        else:
            counts["REACHABLE_MODEL_WRONG"] += 1

    n_joined = sum(counts.values()) - counts["GOLD_NOT_IN_ONTOLOGY_excluded"]
    reach_fail = (counts["UNREACHABLE_pool_suppressed"]
                  + counts["UNREACHABLE_gold_not_in_pool"])
    sel_fail = (counts["REACHABLE_MODEL_WRONG"]
                + counts["REACHABLE_MODEL_DECLINED"])
    n_err = reach_fail + sel_fail

    def pct(a, b):
        return round(a / b * 100, 2) if b else 0.0

    report = {
        "join": "exact (kgp:assesses -> covered_entity_ids); no string match",
        "graph": str(graph.name),
        "primary_arm": args.primary_arm,
        "n_gold_assessments_in_graph": len(gold),
        "n_joined": n_joined,
        "counts": dict(counts.most_common()),
        "n_errors": n_err,
        "reachability_failures": reach_fail,
        "selection_failures": sel_fail,
        "pct_errors_reachability": pct(reach_fail, n_err),
        "pct_errors_selection": pct(sel_fail, n_err),
        "accuracy_of_joined_pct": pct(counts["REACHABLE_MODEL_CORRECT"],
                                      n_joined),
    }
    out_path = Path(args.out) if args.out else (
        res.parent / f"exact_selection_{args.run_id}.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[wrote {out_path}]")


if __name__ == "__main__":
    main()
