#!/usr/bin/env python3
"""Reduce the maximal model to the terms the active questions motivate.

The profile is derived in three steps: a maximal model is built
(`kgprov.ttl`), competency questions are formulated, and the model is
reduced to the terms those questions need. This tool performs the last
step and writes the pruned ontology and a report of what was removed and
why.

The criterion:

  keep a term iff it carries `kgp:motivatedByCQ` naming at least one
  question still in the set.

Removed terms are reported in three groups:

  * **Orphaned**: annotated, but only by questions outside the active
    set. With the default set these are the run-identity terms,
    autonomyLevel and DecodingConfiguration, whose only annotation is
    CQ14.
  * **Unannotated**: they carry no `kgp:motivatedByCQ`, so the criterion
    drops them. --keep-unannotated keeps them instead.
  * **Decided out**: the terms in DECIDED_OUT, removed whatever their
    annotations say. They model decision transfer, the subject of CQ8,
    which is not in DEFAULT_ACTIVE.

Usage:
    python tools/prune_profile.py
    python tools/prune_profile.py --active CQ2,CQ3,CQ6 --keep-unannotated --out out/profile_subset.ttl --report out/profile_subset_report.md

Without --out and --report it rewrites kgprov/ontology/kgprov-profile.ttl
and out/results/prune_report.md. The profile keeps the owl:Ontology
header of kgprov.ttl, IRI and title included.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdflib import Graph, RDF, URIRef  # noqa: E402
from rdflib.namespace import OWL  # noqa: E402

KGP = "https://w3id.org/kgprov#"

#: Questions whose annotations keep a term. CQ1, CQ4, CQ5, CQ7 and CQ8
#: are defined in tools/cq_queries.py but are not in the set, and no term
#: cites them. CQ14 is cited by terms but not defined there, and is not
#: in the set either.
DEFAULT_ACTIVE = ["CQ2", "CQ3", "CQ6", "CQ9", "CQ10", "CQ11", "CQ12", "CQ13"]

#: Annotation properties describing the profile itself. They are not part
#: of the vocabulary being pruned and are always kept.
META = {"motivatedByCQ", "bucket", "captureCost", "module"}

#: Decision-transfer terms (the subject of CQ8, which is not in
#: DEFAULT_ACTIVE), removed whatever their annotations say.
DECIDED_OUT = {
    "DecisionTransfer", "DeduplicationKey", "decidingContext", "decisionTransferredFrom",
}


def local(u: object) -> str:
    return str(u).replace(KGP, "")


class Classification:
    """The reduction, as data rather than as a printed report.

    The single implementation of the criterion: the pruned ontology, the
    report and the tests all read this object.
    """

    def __init__(self, graph: Graph, active: set[str],
                 drop_unannotated: bool = False) -> None:
        self.graph = graph
        self.active = active

        self.kinds: dict[str, str] = {}
        for cls, lbl in ((OWL.Class, "class"), (OWL.ObjectProperty, "objprop"),
                         (OWL.DatatypeProperty, "dataprop"),
                         (OWL.AnnotationProperty, "annprop")):
            for s in graph.subjects(RDF.type, cls):
                if str(s).startswith(KGP):
                    self.kinds.setdefault(local(s), lbl)

        self.motiv: dict[str, set[str]] = defaultdict(set)
        for s, _, o in graph.triples((None, URIRef(KGP + "motivatedByCQ"),
                                      None)):
            self.motiv[local(s)].add(str(o))

        self.annot: dict[str, dict[str, str]] = defaultdict(dict)
        for prop in ("bucket", "captureCost"):
            for s, _, o in graph.triples((None, URIRef(KGP + prop), None)):
                self.annot[local(s)][prop] = str(o)

        # A *term* is a declared class or property. The annotation
        # properties in META describe the profile rather than belonging to
        # it, and enumerated individuals are values of a class rather than
        # terms of their own, so neither is counted.
        self.terms = {t for t in self.kinds if t not in META}

        self.keep: set[str] = set()
        self.orphaned: list[str] = []
        self.unannotated: list[str] = []
        self.decided_out: list[str] = []
        for t in sorted(self.terms):
            if t in DECIDED_OUT:
                self.decided_out.append(t)
            elif t not in self.motiv:
                self.unannotated.append(t)
                if not drop_unannotated:
                    self.keep.add(t)
            elif self.motiv[t] & active:
                self.keep.add(t)
            else:
                self.orphaned.append(t)

        # Enumerated individuals ride with their class.
        self.individuals: set[str] = set()
        for s, _, o in graph.triples((None, RDF.type, None)):
            if str(s).startswith(KGP) and local(o) in self.keep:
                self.individuals.add(local(s))
                self.keep.add(local(s))

    # -- accessors --------------------------------------------------------

    @property
    def retained(self) -> list[str]:
        return sorted(self.keep & self.terms)

    @property
    def removed(self) -> list[str]:
        return sorted(self.terms - self.keep)

    def bucket(self, t: str) -> str:
        return self.annot.get(t, {}).get("bucket", "-")

    def cost(self, t: str) -> str:
        return self.annot.get(t, {}).get("captureCost", "-")

    def cqs(self, t: str) -> list[str]:
        # sort numerically on the digits, so CQ10 follows CQ9 whatever the
        # identifier prefix
        return sorted(self.motiv.get(t, set()) & self.active,
                      key=lambda c: int(re.sub(r"\D", "", c) or 0))

    def by_cq(self) -> dict[str, list[str]]:
        """Active question -> the retained terms it motivates."""
        out: dict[str, list[str]] = defaultdict(list)
        for t in self.retained:
            for cq in self.cqs(t):
                out[cq].append(t)
        return {k: sorted(v) for k, v in out.items()}


def classify(ontology: str | Path = "kgprov/ontology/kgprov.ttl",
             active: set[str] | None = None,
             drop_unannotated: bool = True) -> Classification:
    g = Graph()
    g.parse(str(ROOT / ontology), format="turtle")
    return Classification(g, active or set(DEFAULT_ACTIVE), drop_unannotated)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="kgprov/ontology/kgprov.ttl")
    ap.add_argument("--out", default="kgprov/ontology/kgprov-profile.ttl")
    ap.add_argument("--report", default="out/results/prune_report.md")
    ap.add_argument("--active", default=",".join(DEFAULT_ACTIVE))
    # a term with no kgp:motivatedByCQ annotation is dropped by default;
    # --keep-unannotated keeps such terms instead
    ap.add_argument("--keep-unannotated", action="store_true")
    args = ap.parse_args()

    active = {c.strip() for c in args.active.split(",") if c.strip()}
    c = classify(args.ontology, active, not args.keep_unannotated)
    g, kinds, motiv, terms, keep = c.graph, c.kinds, c.motiv, c.terms, c.keep
    orphaned, unannotated, decided_out = (c.orphaned, c.unannotated,
                                          c.decided_out)

    out = Graph()
    for prefix, ns in g.namespaces():
        out.bind(prefix, ns)
    dropped_triples = 0
    for s, p, o in g:
        subj = local(s)
        if str(s).startswith(KGP) and subj not in keep and subj not in META:
            dropped_triples += 1
            continue
        if isinstance(o, URIRef) and str(o).startswith(KGP):
            obj = local(o)
            if obj not in keep and obj not in META and obj in terms:
                dropped_triples += 1
                continue
        out.add((s, p, o))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=str(out_path), format="turtle")

    lines = [
        "# Pruning report",
        "",
        "Generated by `tools/prune_profile.py`. The criterion: keep a term",
        "iff it carries `kgp:motivatedByCQ` naming at least one question",
        "still in the set.",
        "",
        f"- Active questions: `{'`, `'.join(sorted(active))}`",
        f"- Maximal model: **{len(terms)}** terms",
        f"- Retained: **{len(keep & terms)}**",
        f"- Removed: **{len(terms) - len(keep & terms)}** "
        f"({len(orphaned)} orphaned, {len(decided_out)} decided out, "
        f"{len(unannotated) if (not args.keep_unannotated) else 0} unannotated)",
        f"- Triples dropped: {dropped_triples}",
        "",
        "## Orphaned — annotated only by questions no longer in the set",
        "",
        "Removing a question strands the terms only it motivated. This is a",
        "cost of that decision and is recorded rather than absorbed.",
        "",
    ]
    for t in orphaned:
        lines.append(f"- `{t}` ({kinds.get(t,'?')}) — "
                     f"motivated by {', '.join(sorted(motiv[t]))}")
    lines += [
        "",
        "## Decided out — removed with the question that motivated them",
        "",
        "The decision-transfer terms in DECIDED_OUT are removed whatever",
        "their annotations say. They model decision transfer, the subject of CQ8.",
        "",
    ]
    for t in decided_out:
        lines.append(f"- `{t}` ({kinds.get(t,'?')})")
    lines += [
        "",
        "## Unannotated — the criterion cannot judge these",
        "",
        ("**Dropped.** No question in the active set reaches these, so the "
         "criterion removes them like any other unreached term."
         if (not args.keep_unannotated) else
         "**Retained**, because `--keep-unannotated` was given. The default "
         "drops them."),
        "",
    ]
    for t in unannotated:
        lines.append(f"- `{t}` ({kinds.get(t,'?')})")

    rep_path = ROOT / args.report
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"maximal {len(terms)} -> retained {len(keep & terms)} "
          f"({len(orphaned)} orphaned, {len(decided_out)} decided out, "
          f"{len(unannotated)} unannotated"
          f"{' dropped' if (not args.keep_unannotated) else ' kept'})")
    print(f"  profile: {out_path}")
    print(f"  report : {rep_path}")


if __name__ == "__main__":
    main()
