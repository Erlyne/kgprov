#!/usr/bin/env python3
"""Check a provenance graph against the KGPROV class shapes.

The shapes say how the profile's classes relate through core PROV: a
narrowing activity used a constraint space, a grounded mention was quoted
from a target chunk. They are SHACL rather than OWL restrictions because
``owl:allValuesFrom`` is an inference rule, not a check. Given

    <narrow_42> a kgp:NarrowingActivity ; prov:used <decoding_config> .

an OWL reasoner does not reject the graph; under the open-world
assumption it concludes ``<decoding_config> a kgp:ConstraintSpace``.
As a SHACL shape the same statement is a constraint, and the graph above
is a violation.

This runs the shapes without adding a dependency: each ``sh:property``
with an ``sh:path`` and an ``sh:class`` compiles to one SPARQL query for
the value nodes that fail it. Subclass closure is taken from the ontology
(``rdf:type/rdfs:subClassOf*``), which is what SHACL's ``sh:class``
means. Every constraint space is typed ``DerivedConstraintSpace`` or
``StaticConstraintSpace`` as well as ``ConstraintSpace``, so a bare type
check would pass on these graphs without testing the subclass relation.

``kgprov-shapes.ttl`` is a standard SHACL shapes graph, so a full SHACL
processor can be used instead; this module lets the test suite run the
shapes without one.

Usage:
    python tools/validate_shapes.py
    python tools/validate_shapes.py --graph out/provenance/stage1_weak_clean.ttl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdflib import Graph, Namespace, RDF  # noqa: E402

SH = Namespace("http://www.w3.org/ns/shacl#")

PREFIXES = """
PREFIX kgp:  <https://w3id.org/kgprov#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


@dataclass(frozen=True)
class Shape:
    """One sh:property constraint, flattened."""
    target: str          # sh:targetClass
    path: str            # sh:path
    cls: str             # sh:class
    message: str

    def query(self) -> str:
        """Value nodes of `path` on a `target` that are not a `cls`."""
        return f"""{PREFIXES}
SELECT ?node ?value WHERE {{
  ?node rdf:type/rdfs:subClassOf* <{self.target}> .
  ?node <{self.path}> ?value .
  FILTER NOT EXISTS {{
    ?value rdf:type/rdfs:subClassOf* <{self.cls}> .
  }}
}}"""


@dataclass
class Violation:
    shape: Shape
    node: str
    value: str

    def __str__(self) -> str:
        return (f"{self.shape.message}\n"
                f"    node  : {self.node}\n"
                f"    value : {self.value}\n"
                f"    path  : {self.shape.path}")


def load_shapes(path: Path) -> list[Shape]:
    g = Graph()
    g.parse(str(path), format="turtle")
    out: list[Shape] = []
    for shape in g.subjects(RDF.type, SH.NodeShape):
        targets = list(g.objects(shape, SH.targetClass))
        for prop in g.objects(shape, SH.property):
            paths = list(g.objects(prop, SH.path))
            classes = list(g.objects(prop, SH["class"]))
            msgs = list(g.objects(prop, SH.message))
            if not (targets and paths and classes):
                continue
            out.append(Shape(
                target=str(targets[0]), path=str(paths[0]),
                cls=str(classes[0]),
                message=str(msgs[0]) if msgs else
                f"{targets[0]} {paths[0]} must be {classes[0]}"))
    return sorted(out, key=lambda s: (s.target, s.path))


def validate(data: Graph, shapes: list[Shape],
             limit: int = 5) -> list[Violation]:
    """Every violation, capped per shape so one systematic fault cannot
    bury the rest of the report."""
    found: list[Violation] = []
    for sh in shapes:
        seen = 0
        for row in data.query(sh.query()):
            found.append(Violation(sh, str(row.node), str(row.value)))
            seen += 1
            if seen >= limit:
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="out/provenance/stage1_weak_clean.ttl")
    ap.add_argument("--shapes", default="kgprov/ontology/kgprov-shapes.ttl")
    ap.add_argument("--ontology", default="kgprov/ontology/kgprov.ttl",
                    help="Loaded for rdfs:subClassOf closure. sh:class is "
                         "defined over subclasses, and the run graph does "
                         "not carry the vocabulary's own axioms.")
    args = ap.parse_args()

    shapes = load_shapes(ROOT / args.shapes)
    print(f"[shapes] {len(shapes)} constraints from "
          f"{Path(args.shapes).name}")

    gp = ROOT / args.graph
    if not gp.exists():
        print(f"[shapes] graph not found: {gp}", file=sys.stderr)
        return 2

    data = Graph()
    print(f"[shapes] loading {gp.name} ...")
    data.parse(str(gp), format="turtle")
    data.parse(str(ROOT / args.ontology), format="turtle")
    print(f"[shapes] {len(data):,} triples (graph + vocabulary)")

    violations = validate(data, shapes)
    if not violations:
        print(f"[shapes] conforms: 0 violations over {len(shapes)} "
              f"constraints")
        return 0

    print(f"\n[shapes] {len(violations)} violation(s):\n")
    for v in violations:
        print(f"  * {v}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
