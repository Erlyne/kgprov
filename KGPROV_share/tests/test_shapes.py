"""The class shapes must reject a bad graph, not repair it.

An OWL restriction of the form

    kgp:NarrowingActivity rdfs:subClassOf
        [ a owl:Restriction ; owl:onProperty prov:used ;
          owl:allValuesFrom kgp:ConstraintSpace ] .

states the shape without constraining prov:used globally, but under the
open-world assumption owl:allValuesFrom concludes rather than checks:
given a narrowing activity recorded as using a decoding configuration, a
reasoner derives that the configuration is a ConstraintSpace. The SHACL
shapes report the same graph as a violation;
test_the_owl_encoding_would_have_inferred_instead_of_rejected shows both.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, Namespace, RDF, URIRef

from tools.validate_shapes import load_shapes, validate

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ROOT / "kgprov" / "ontology" / "kgprov-shapes.ttl"
ONTOLOGY = ROOT / "kgprov" / "ontology" / "kgprov.ttl"

KGP = Namespace("https://w3id.org/kgprov#")
PROV = Namespace("http://www.w3.org/ns/prov#")
EX = Namespace("http://example.org/run#")


@pytest.fixture(scope="module")
def shapes():
    return load_shapes(SHAPES)


@pytest.fixture
def vocab():
    """The vocabulary alone, for rdfs:subClassOf closure."""
    g = Graph()
    g.parse(str(ONTOLOGY), format="turtle")
    return g


def _narrowing(g: Graph, used) -> Graph:
    """A narrowing activity that used `used`, and nothing else."""
    g.add((EX.narrow_42, RDF.type, KGP.NarrowingActivity))
    g.add((EX.narrow_42, PROV.used, used))
    return g


# ---------------------------------------------------------------------
#  Loading the shapes
# ---------------------------------------------------------------------

def test_shape_count(shapes):
    """Sixteen sh:property constraints. Three of them narrow a core
    property to a domain class for this pipeline: kgp:assesses,
    kgp:decisionTransferredFrom and kgp:decidingContext."""
    assert len(shapes) == 16


def test_every_shape_carries_a_message(shapes):
    for s in shapes:
        assert s.message and not s.message.startswith("https://")


def test_ontology_no_longer_declares_value_restrictions(vocab):
    """The ontology declares no owl:Restriction and no owl:allValuesFrom."""
    OWL = Namespace("http://www.w3.org/2002/07/owl#")
    assert not list(vocab.subjects(RDF.type, OWL.Restriction))
    assert not list(vocab.triples((None, OWL.allValuesFrom, None)))


# ---------------------------------------------------------------------
#  The counterexample
# ---------------------------------------------------------------------

def test_narrowing_that_used_a_config_parameter_is_a_violation(
        shapes, vocab):
    """A narrowing activity that used a decoding parameter is a violation.

    A narrowing activity consumes the space it reduces; recording it as
    using a decoding parameter is a capture defect.
    """
    g = _narrowing(vocab, EX.decoding_config)
    g.add((EX.decoding_config, RDF.type, KGP.DecodingConfiguration))

    violations = validate(g, shapes)
    assert len(violations) == 1
    v = violations[0]
    assert v.node == str(EX.narrow_42)
    assert v.value == str(EX.decoding_config)
    assert v.shape.path == str(PROV.used)
    assert v.shape.cls == str(KGP.ConstraintSpace)


def test_the_owl_encoding_would_have_inferred_instead_of_rejected(vocab):
    """OWL entails what SHACL rejects.

    Under owl:allValuesFrom the offending value is entailed to be a
    ConstraintSpace, so the graph is consistent and a query for
    constraint spaces returns a decoding parameter. The test simulates
    that entailment, then checks that the SHACL shapes reject the graph.
    """
    g = _narrowing(vocab, EX.decoding_config)
    g.add((EX.decoding_config, RDF.type, KGP.DecodingConfiguration))

    # What a reasoner does with allValuesFrom on prov:used:
    inferred = Graph()
    for t in g:
        inferred.add(t)
    for _, _, value in g.triples((EX.narrow_42, PROV.used, None)):
        inferred.add((value, RDF.type, KGP.ConstraintSpace))

    spaces = set(inferred.subjects(RDF.type, KGP.ConstraintSpace))
    assert EX.decoding_config in spaces, (
        "this is the defect: the parameter is now a constraint space")

    # And under SHACL the same graph is rejected instead.
    assert validate(g, load_shapes(SHAPES))


def test_a_correct_narrowing_conforms(shapes, vocab):
    g = _narrowing(vocab, EX.space_7)
    g.add((EX.space_7, RDF.type, KGP.ConstraintSpace))
    assert validate(g, shapes) == []


def test_sh_class_honours_subclass_closure(shapes, vocab):
    """Spaces are typed by their subclass; a bare type check would pass
    only by accident, and would fail on a graph that typed them
    precisely."""
    g = _narrowing(vocab, EX.space_7)
    g.add((EX.space_7, RDF.type, KGP.DerivedConstraintSpace))
    assert (KGP.DerivedConstraintSpace, URIRef(
        "http://www.w3.org/2000/01/rdf-schema#subClassOf"),
        KGP.ConstraintSpace) in vocab
    assert validate(g, shapes) == []


def test_attestation_used_is_deliberately_unconstrained(shapes, vocab):
    """No shape constrains prov:used on an Attestation.

    The recorder writes prov:used from an attestation both to the claim it
    checks (Recorder.attestation) and to the registry snapshot it names
    with kgp:attestedAgainst (_Attestation.against), so prov:used here has
    two kinds of object and is left unconstrained.
    """
    g = vocab
    g.add((EX.att_1, RDF.type, KGP.Attestation))
    g.add((EX.att_1, PROV.used, EX.some_claim))
    g.add((EX.att_1, PROV.wasAssociatedWith, EX.agent))
    g.add((EX.agent, RDF.type, KGP.EscalationSearchAgent))

    paths = {(s.target, s.path) for s in shapes}
    assert (str(KGP.Attestation), str(PROV.used)) not in paths
    assert validate(g, shapes) == []
