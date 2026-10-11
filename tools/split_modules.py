#!/usr/bin/env python3
"""Split kgprov.ttl into a domain-neutral core and a biomedical module.

Checks that the core vocabulary (ConstraintSpace, wasConstrainedBy,
NarrowingActivity, Attestation and the rest) does not depend on the
biomedical terms, then writes the two modules as separate files.

Each term in `kgprov.ttl` carries `kgp:module "core"` or
`kgp:module "domain"`. Core is the constraint, narrowing and attestation
machinery. Domain is what this particular pipeline manipulates
(manuscripts, chunks, spans, descriptors, predicates, its five software
agents): the part a pipeline outside biomedicine would replace, and
which `prov:type` could carry instead.

The invariant: a core term may not refer to a domain term through
`rdfs:subClassOf`, `rdfs:subPropertyOf`, `rdfs:domain` or `rdfs:range`.
This one-way dependency is what lets the core be imported without the
domain module; splitting the terms into two files does not by itself
guarantee that.

The layering follows the Research Object suite, where `ro` is
domain-agnostic and `wfdesc` and `wfprov` sit above it for specification
and execution traces respectively.

The four emitted files (two modules and their two shapes graphs) are
derived artefacts; `kgprov.ttl` remains the canonical, commented source.
Each module also carries the `owl:Ontology` header of `kgprov.ttl` next
to its own, and uses the annotation properties `kgp:motivatedByCQ`,
`kgp:bucket`, `kgp:captureCost` and `kgp:module` without declaring them.

Usage:
    python tools/split_modules.py             # rewrites the four files in kgprov/ontology/
    python tools/split_modules.py --check     # invariant only, no writes
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdflib import Graph, RDF, RDFS, URIRef  # noqa: E402
from rdflib.namespace import OWL  # noqa: E402

KGP = "https://w3id.org/kgprov#"
CORE_IRI = URIRef("https://w3id.org/kgprov/core")
DOMAIN_IRI = URIRef("https://w3id.org/kgprov/bio")

#: Links along which a core term must not reach a domain term.
STRUCTURAL = (RDFS.subClassOf, RDFS.subPropertyOf, RDFS.domain, RDFS.range)


def local(u: object) -> str:
    return str(u).replace(KGP, "")


def modules(g: Graph) -> dict[str, str]:
    out = {}
    for s, _, o in g.triples((None, URIRef(KGP + "module"), None)):
        out[local(s)] = str(o)
    return out


def check(g: Graph, mods: dict[str, str]) -> list[str]:
    """Core terms that reach a domain term. Empty list means the core
    module stands alone."""
    bad = []
    for term, mod in sorted(mods.items()):
        if mod != "core":
            continue
        subj = URIRef(KGP + term)
        for link in STRUCTURAL:
            for o in g.objects(subj, link):
                if not str(o).startswith(KGP):
                    continue
                if mods.get(local(o)) == "domain":
                    bad.append(
                        f"{term} --{local(link)}--> {local(o)} (domain)")
    return bad


def emit(g: Graph, mods: dict[str, str], want: str,
         onto_iri: URIRef, imports: URIRef | None) -> Graph:
    keep = {t for t, m in mods.items() if m == want}
    # Enumerated individuals ride with the class that declares them.
    for s, _, o in g.triples((None, RDF.type, None)):
        if str(s).startswith(KGP) and local(o) in keep:
            keep.add(local(s))

    out = Graph()
    for prefix, ns in g.namespaces():
        out.bind(prefix, ns)
    for s, p, o in g:
        if str(s).startswith(KGP) and local(s) not in keep:
            continue
        # Do not carry a statement whose object this module lacks.
        if (isinstance(o, URIRef) and str(o).startswith(KGP)
                and local(o) in mods and local(o) not in keep):
            continue
        out.add((s, p, o))

    out.add((onto_iri, RDF.type, OWL.Ontology))
    out.add((onto_iri, OWL.imports,
             URIRef("http://www.w3.org/ns/prov-o#")))
    if imports is not None:
        out.add((onto_iri, OWL.imports, imports))
    return out


def split_shapes(shapes: Graph, mods: dict[str, str]) -> tuple[Graph, Graph]:
    """Partition the shapes graph by the same rule.

    A shape belongs to the domain module if it targets a domain class or
    constrains a value to one -- including shapes on *core* classes, of
    which there are four: a narrowing activity is associated with this
    pipeline's evaluator agent, an attestation with its escalation agent,
    a gold annotation assesses a mention, and a decision transfer's
    deciding context is a mention. Each narrows a core class
    to a domain one, so each is a statement about this pipeline rather
    than about the profile, and each would be wrong for a consumer that
    imported the core alone.
    """
    SH = URIRef("http://www.w3.org/ns/shacl#")
    sh_target = URIRef(str(SH) + "targetClass")
    sh_property = URIRef(str(SH) + "property")
    sh_class = URIRef(str(SH) + "class")

    def is_domain(shape) -> bool:
        refs = list(shapes.objects(shape, sh_target))
        for prop in shapes.objects(shape, sh_property):
            refs += list(shapes.objects(prop, sh_class))
        return any(mods.get(local(r)) == "domain" for r in refs)

    core, dom = Graph(), Graph()
    for out in (core, dom):
        for prefix, ns in shapes.namespaces():
            out.bind(prefix, ns)

    node_shapes = set(shapes.subjects(sh_target, None))
    for shape in node_shapes:
        target = dom if is_domain(shape) else core
        for p, o in shapes.predicate_objects(shape):
            target.add((shape, p, o))
            # carry the property-shape blank nodes with their shape
            if p == sh_property:
                for pp, oo in shapes.predicate_objects(o):
                    target.add((o, pp, oo))
    return core, dom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="kgprov/ontology/kgprov.ttl")
    ap.add_argument("--core", default="kgprov/ontology/kgprov-core.ttl")
    ap.add_argument("--domain", default="kgprov/ontology/kgprov-bio.ttl")
    ap.add_argument("--shapes", default="kgprov/ontology/kgprov-shapes.ttl")
    ap.add_argument("--check", action="store_true",
                    help="Verify the invariant and write nothing.")
    args = ap.parse_args()

    g = Graph()
    g.parse(str(ROOT / args.ontology), format="turtle")
    mods = modules(g)

    counts = defaultdict(int)
    for m in mods.values():
        counts[m] += 1
    print(f"modules: core {counts['core']}, domain {counts['domain']} "
          f"({len(mods)} terms)")

    unassigned = [t for t in {local(s) for s in g.subjects(RDF.type, None)
                              if str(s).startswith(KGP)}
                  if t not in mods
                  and (URIRef(KGP + t), RDF.type, OWL.Class) in g]
    if unassigned:
        print(f"\n!! {len(unassigned)} class(es) carry no kgp:module: "
              f"{', '.join(sorted(unassigned))}", file=sys.stderr)
        return 1

    violations = check(g, mods)
    if violations:
        print(f"\n!! the core module does not stand alone --- "
              f"{len(violations)} reference(s) into the domain module:\n",
              file=sys.stderr)
        for v in violations:
            print(f"   {v}", file=sys.stderr)
        print("\n   Either the term is domain and mislabelled, or the link "
              "\n   should be relaxed to a core/prov: class and narrowed by "
              "\n   a shape in the domain module.", file=sys.stderr)
        return 1
    print("invariant holds: no core term refers to a domain term")

    if args.check:
        return 0

    core = emit(g, mods, "core", CORE_IRI, None)
    dom = emit(g, mods, "domain", DOMAIN_IRI, CORE_IRI)
    written = [(core, args.core), (dom, args.domain)]

    shapes_path = ROOT / args.shapes
    if shapes_path.exists():
        shapes = Graph()
        shapes.parse(str(shapes_path), format="turtle")
        core_sh, dom_sh = split_shapes(shapes, mods)
        written += [
            (core_sh, args.core.replace(".ttl", "-shapes.ttl")),
            (dom_sh, args.domain.replace(".ttl", "-shapes.ttl")),
        ]

    for graph, path in written:
        p = ROOT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=str(p), format="turtle")
        print(f"  {path}: {len(graph)} triples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
