"""The core module must stand alone.

No core term may refer to a domain term through rdfs:subClassOf,
rdfs:subPropertyOf, rdfs:domain or rdfs:range, so the constraint,
narrowing and attestation vocabulary can be imported without this
pipeline's biomedical classes. These tests pin that one-way dependency,
the module counts, and the matching split of the shapes graph.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, Namespace, RDF, RDFS, URIRef

from tools.split_modules import check, modules, split_shapes

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "kgprov" / "ontology" / "kgprov.ttl"
SHAPES = ROOT / "kgprov" / "ontology" / "kgprov-shapes.ttl"

KGP = Namespace("https://w3id.org/kgprov#")
PROV = Namespace("http://www.w3.org/ns/prov#")
SH = Namespace("http://www.w3.org/ns/shacl#")


@pytest.fixture(scope="module")
def g():
    graph = Graph()
    graph.parse(str(ONTOLOGY), format="turtle")
    return graph


@pytest.fixture(scope="module")
def mods(g):
    return modules(g)


# ---------------------------------------------------------------------
#  Every term is assigned, once
# ---------------------------------------------------------------------

def test_every_term_carries_a_module(g, mods):
    from tools.prune_profile import classify
    assert set(mods) == set(classify().terms)
    assert set(mods.values()) == {"core", "domain"}


def test_the_split_is_61_core_and_26_domain(mods):
    """kgp:cuiValue and kgp:mappingTier count as domain terms: both hang
    off kgp:CanonicalConceptID, which is a domain class."""
    counts = {m: sum(1 for v in mods.values() if v == m)
              for m in ("core", "domain")}
    assert counts == {"core": 61, "domain": 26}


# ---------------------------------------------------------------------
#  The invariant
# ---------------------------------------------------------------------

def test_no_core_term_refers_to_a_domain_term(g, mods):
    """If this fails, the core cannot be imported alone."""
    assert check(g, mods) == []


def test_the_invariant_would_catch_a_regression(g, mods):
    """Giving kgp:assesses a domain range is reported as a violation."""
    broken = Graph()
    for t in g:
        broken.add(t)
    broken.add((KGP.assesses, RDFS.range, KGP.GroundedMention))

    violations = check(broken, mods)
    assert violations
    assert any("assesses" in v and "GroundedMention" in v
               for v in violations)


def test_the_structural_terms_are_core(mods):
    """The constraint, narrowing and attestation vocabulary is core."""
    for t in ("ConstraintSpace", "DerivedConstraintSpace", "SpaceMembership",
              "NarrowingActivity", "narrowedFrom", "excluded",
              "exclusionReason", "wasConstrainedBy", "constraintEnforcement",
              "enforcementArtifact", "Attestation", "attestedAgainst",
              "attestationOutcome", "isRepeatable"):
        assert mods[t] == "core", f"{t} should be domain-neutral"


def test_the_domain_nouns_and_agents_are_domain(mods):
    """Pipeline-specific nouns and agents, which prov:type could carry."""
    for t in ("SourceManuscript", "TargetChunk", "RawSpan",
              "GroundedMention", "CanonicalConceptID", "KnowledgeGraphEdge",
              "ExtractionAgent", "PrimaryEvaluatorAgent",
              "EscalationSearchAgent", "RelationExtractionAgent",
              "NormalizationAgent", "semanticCategory"):
        assert mods[t] == "domain", f"{t} is not domain-neutral"


def test_no_core_term_mentions_biomedicine(g, mods):
    """A blunt lexical check on the labels and comments, which the
    structural invariant cannot see."""
    banned = ("mesh", "biomedical", "chemical", "disease", "pubmed",
              "descriptor", "manuscript")
    offenders = []
    for term, mod in mods.items():
        if mod != "core":
            continue
        subj = URIRef(str(KGP) + term)
        text = " ".join(
            str(o).lower()
            for p in (RDFS.label, RDFS.comment)
            for o in g.objects(subj, p))
        hits = [w for w in banned if w in text]
        if hits:
            offenders.append(f"{term}: {hits}")
    assert not offenders, "core terms naming the domain: " + "; ".join(
        offenders)


# ---------------------------------------------------------------------
#  Shapes partition by the same rule
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def shape_graphs(mods):
    shapes = Graph()
    shapes.parse(str(SHAPES), format="turtle")
    return split_shapes(shapes, mods)


def test_core_shapes_never_reference_a_domain_class(shape_graphs, mods):
    core_sh, _ = shape_graphs
    for _, _, o in core_sh:
        if isinstance(o, URIRef) and str(o).startswith(str(KGP)):
            name = str(o).replace(str(KGP), "")
            assert mods.get(name) != "domain", (
                f"core shape graph references domain term {name}")


def test_shapes_narrowing_core_classes_land_in_the_domain_module(
        shape_graphs):
    """Four shapes target a core class and constrain it to a domain one.
    This test checks three: the evaluator agent, the escalation agent,
    and gold assessing a mention. Each is a statement about this
    pipeline, not the profile."""
    _, dom_sh = shape_graphs
    targets = {str(o) for _, _, o in dom_sh.triples((None, SH.targetClass,
                                                     None))}
    for t in ("NarrowingActivity", "Attestation", "GoldAnnotation"):
        assert str(KGP) + t in targets, (
            f"{t} shape narrows to a domain class and belongs in the "
            f"domain shapes graph")
