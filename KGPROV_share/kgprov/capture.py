"""
The provenance capture sidecar.

Capture is orthogonal to pipeline logic: nothing in this module knows
what a mention or a CUI is. A phase announces what it is about to do,
does it, and announces what came out. The recorder turns that into RDF
conforming to kgprov.ttl.

Two consequences:

  * Provenance capture can be bolted onto a generative pipeline without
    entangling it, which is what makes the profile portable to pipelines
    other than this one.
  * Changing the pipeline does not change the provenance vocabulary. The
    vocabulary changes only when the set of claims the system can make
    changes.

Everything emitted here is valid W3C PROV. A consumer that ignores the
kgprov namespace still reads a coherent, if impoverished, graph.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD

from . import ids

KGP = Namespace("https://w3id.org/kgprov#")
PROV = Namespace("http://www.w3.org/ns/prov#")

#: Namespace for instance data produced by a run. Change once, here.
BASE = Namespace("https://w3id.org/kgprov/run/")


def _now() -> Literal:
    return Literal(dt.datetime.now(dt.timezone.utc).isoformat(), datatype=XSD.dateTime)


#: Characters the Turtle grammar forbids inside <...>. rdflib's serializer
#: does not enforce this and will emit them; a strict parser then refuses
#: the whole file. Kept identical to `tools/sanitize_ttl.py` so a sanitised
#: graph and a freshly written one agree on IRIs.
_IRI_FORBIDDEN = set(' <>{}|^`"[]')


def iri(local_name: str) -> URIRef:
    """Instance IRI from a content-addressed identifier.

    Percent-encodes everything Turtle forbids. Local names are built from
    corpus text (dedup keys, resolved strings, CUI values), and biomedical
    nomenclature often contains brackets, e.g. ``[3H]spiperone``. rdflib
    writes such IRIs without complaint, but a strict parser such as
    pyoxigraph rejects the whole file.
    """
    safe = local_name.replace(":", "_").replace(" ", "_")
    if any(c in _IRI_FORBIDDEN or ord(c) < 0x20 for c in safe):
        safe = "".join(
            "%%%02X" % ord(c)
            if (c in _IRI_FORBIDDEN or ord(c) < 0x20) else c
            for c in safe)
    return BASE[safe]


# ---------------------------------------------------------------------
#  Constraint spaces
# ---------------------------------------------------------------------

@dataclass
class SpaceMember:
    """One candidate in a constraint space.

    ``score`` records how close a non-selected member came, which is what
    turns CQ2 from a binary reachability check into a measurement.
    ``probability_mass`` is populated only where the engine exposes
    logits; it stays None for API models and that absence is itself
    informative.
    """
    value: str
    rank: int | None = None
    score: float | None = None
    probability_mass: float | None = None
    selected: bool = False
    label: str | None = None


#: How an admissible set was applied to a generation. The distinction is
#: the profile's, not PROV's: PROV records that an activity used a plan but
#: has no predicate for whether the plan's enumeration *bound* generation or
#: merely described it. Under DECODE_TIME an excluded value could not have
#: been produced; under POST_VALIDATED it could have been, and was rejected
#: after the fact. That decides whether a rejection count is a true zero or
#: a lower bound.
ENFORCE_DECODE_TIME = "DECODE_TIME"
ENFORCE_POST_VALIDATED = "POST_VALIDATED"
ENFORCE_UNENFORCED = "UNENFORCED"
ENFORCEMENT_MODES = frozenset(
    {ENFORCE_DECODE_TIME, ENFORCE_POST_VALIDATED, ENFORCE_UNENFORCED})


@dataclass
class ConstraintSpace:
    """The enumerated set of values an activity was permitted to produce.

    ``derived`` distinguishes a space recoverable from the plan (e.g. the
    Phase 1 category whitelist) from one computed at runtime out of
    upstream results. The two are typed kgp:StaticConstraintSpace and
    kgp:DerivedConstraintSpace in the graph.
    """
    kind: str
    members: list[SpaceMember]
    derived: bool = True
    narrowed_from: str | None = None
    discriminators: tuple = ()
    uri_ref: URIRef | None = field(default=None, repr=False)

    @property
    def space_id(self) -> str:
        return ids.space_id(self.kind, *self.discriminators)

    @property
    def values(self) -> list[str]:
        return [m.value for m in self.members]

    def contains(self, value: str) -> bool:
        return any(m.value == value for m in self.members)


def space(kind: str, values: Iterable[Any], *, derived: bool = True,
          narrowed_from: str | None = None,
          discriminators: Sequence[Any] = (),
          scores: dict[str, float] | None = None,
          selected: str | None = None) -> ConstraintSpace:
    """Convenience constructor for the common case."""
    members = []
    for i, v in enumerate(values):
        v = str(v)
        members.append(SpaceMember(
            value=v,
            rank=i + 1,
            score=None if scores is None else scores.get(v),
            selected=(v == selected),
        ))
    return ConstraintSpace(kind=kind, members=members, derived=derived,
                           narrowed_from=narrowed_from,
                           discriminators=tuple(discriminators))


# ---------------------------------------------------------------------
#  Recorder
# ---------------------------------------------------------------------

class Recorder:
    """Accumulates a provenance graph for one pipeline run."""

    def __init__(self, run_id: str, *, config_variant: str = "default",
                 model_id: str = "unknown", ontology_version: str = "unknown",
                 pipeline_version: str = "0.1.0"):
        self.run_id = run_id
        self.g = Graph()
        self.g.bind("kgp", KGP)
        self.g.bind("prov", PROV)
        self.g.bind("run", BASE)

        self.run_uri = iri(f"run_{run_id}")
        self.config_variant = config_variant
        self.model_id = model_id
        self.ontology_version = ontology_version

        g = self.g
        g.add((self.run_uri, RDF.type, KGP.PipelineRun))
        g.add((self.run_uri, RDF.type, PROV.Bundle))
        g.add((self.run_uri, PROV.generatedAtTime, _now()))
        g.add((self.run_uri, KGP.configurationVariantLabel, Literal(config_variant)))
        g.add((self.run_uri, KGP.modelId, Literal(model_id)))
        g.add((self.run_uri, KGP.ontologyVersion, Literal(ontology_version)))
        g.add((self.run_uri, KGP.pipelineVersion, Literal(pipeline_version)))

        self._spaces: dict[str, URIRef] = {}

    # -- entities ----------------------------------------------------

    def entity(self, local_id: str, cls: URIRef | None = None,
               label: str | None = None, **attrs) -> URIRef:
        u = iri(local_id)
        self.g.add((u, RDF.type, PROV.Entity))
        if cls is not None:
            self.g.add((u, RDF.type, cls))
        if label is not None:
            self.g.add((u, RDFS.label, Literal(label)))
        for k, v in attrs.items():
            if v is not None:
                self.g.add((u, KGP[k], Literal(v)))
        return u

    def agent(self, local_id: str, cls: URIRef | None = None,
              on_behalf_of: str | None = None) -> URIRef:
        u = iri(local_id)
        self.g.add((u, RDF.type, PROV.SoftwareAgent))
        if cls is not None:
            self.g.add((u, RDF.type, cls))
        if on_behalf_of:
            self.g.add((u, PROV.actedOnBehalfOf, iri(on_behalf_of)))
        return u

    def quoted_from(self, derived: str, source: str) -> None:
        """Verbatim text is quoted, not merely derived. PROV distinguishes these."""
        self.g.add((iri(derived), PROV.wasQuotedFrom, iri(source)))

    def primary_source(self, derived: str, source: str) -> None:
        self.g.add((iri(derived), PROV.hadPrimarySource, iri(source)))

    def revision_of(self, new: str, old: str) -> None:
        self.g.add((iri(new), PROV.wasRevisionOf, iri(old)))

    def invalidate(self, local_id: str, reason: URIRef,
                   by_activity: str | None = None) -> None:
        """Record that an entity did not survive. CQ9 reads this."""
        u = iri(local_id)
        inv = BNode()
        self.g.add((u, PROV.qualifiedInvalidation, inv))
        self.g.add((inv, RDF.type, PROV.Invalidation))
        self.g.add((inv, KGP.hadRejectionReason, reason))
        self.g.add((inv, PROV.atTime, _now()))
        if by_activity:
            self.g.add((inv, PROV.activity, iri(by_activity)))
            self.g.add((u, PROV.wasInvalidatedBy, iri(by_activity)))

    # -- constraint spaces -------------------------------------------

    def record_space(self, cs: ConstraintSpace) -> URIRef:
        sid = cs.space_id
        if sid in self._spaces:
            return self._spaces[sid]

        u = iri(sid)
        g = self.g
        g.add((u, RDF.type, KGP.ConstraintSpace))
        g.add((u, RDF.type, PROV.Collection))
        g.add((u, RDF.type,
               KGP.DerivedConstraintSpace if cs.derived else KGP.StaticConstraintSpace))
        g.add((u, KGP.spaceKind, Literal(cs.kind)))
        g.add((u, KGP.cardinality, Literal(len(cs.members), datatype=XSD.integer)))

        if cs.narrowed_from:
            g.add((u, KGP.narrowedFrom, iri(cs.narrowed_from)))
            g.add((u, PROV.wasDerivedFrom, iri(cs.narrowed_from)))

        for m in cs.members:
            mu = iri(f"{sid}_m_{ids.content_hash(m.value)[:8]}")
            g.add((mu, RDF.type, PROV.Entity))
            g.add((mu, RDFS.label, Literal(m.label or m.value)))
            g.add((mu, KGP.memberValue, Literal(m.value)))
            # plain membership, readable by a vanilla PROV consumer
            g.add((u, PROV.hadMember, mu))
            # qualified membership, carrying rank / score / selection
            qm = BNode()
            g.add((u, KGP.qualifiedMembership, qm))
            g.add((qm, RDF.type, KGP.SpaceMembership))
            g.add((qm, KGP.member, mu))
            g.add((qm, KGP.wasSelected, Literal(m.selected, datatype=XSD.boolean)))
            if m.rank is not None:
                g.add((qm, KGP.rank, Literal(m.rank, datatype=XSD.integer)))
            if m.score is not None:
                g.add((qm, KGP.score, Literal(m.score, datatype=XSD.double)))
            if m.probability_mass is not None:
                g.add((qm, KGP.probabilityMass,
                       Literal(m.probability_mass, datatype=XSD.double)))

        self._spaces[sid] = u
        return u

    # -- activities --------------------------------------------------

    @contextmanager
    def generation(self, phase: str, *, agent: str, activity_cls: URIRef | None = None,
                   plan: str | None = None, prompt_hash: str | None = None,
                   autonomy_level: int | None = None,
                   discriminators: Sequence[Any] = ()) -> Iterator["_Call"]:
        """Wrap one constrained generative call."""
        aid = ids.activity_id(phase, *discriminators)
        u = iri(aid)
        g = self.g
        g.add((u, RDF.type, PROV.Activity))
        if activity_cls is not None:
            g.add((u, RDF.type, activity_cls))
        g.add((u, PROV.startedAtTime, _now()))
        g.add((u, KGP.phase, Literal(phase)))
        if autonomy_level is not None:
            g.add((u, KGP.autonomyLevel, Literal(autonomy_level, datatype=XSD.integer)))

        # The qualified association carries the agent. The prompt is a
        # prov:Entity the activity used (prov:used), following the
        # PROV-AGENT mapping.
        assoc = BNode()
        g.add((u, PROV.qualifiedAssociation, assoc))
        g.add((assoc, RDF.type, PROV.Association))
        g.add((assoc, PROV.agent, iri(agent)))
        g.add((u, PROV.wasAssociatedWith, iri(agent)))
        if plan:
            pu = iri(plan)
            g.add((pu, RDF.type, PROV.Entity))
            g.add((pu, RDF.type, KGP.PromptTemplate))
            if prompt_hash:
                g.add((pu, KGP.promptHash, Literal(prompt_hash)))
            g.add((u, PROV.used, pu))

        call = _Call(self, u, aid)
        try:
            yield call
        finally:
            g.add((u, PROV.endedAtTime, _now()))

    @contextmanager
    def narrowing(self, stage: str, *, input_space: ConstraintSpace,
                  agent: str | None = None, gold: str | None = None,
                  discriminators: Sequence[Any] = ()) -> Iterator["_Narrowing"]:
        """Wrap one space-reduction step in the retrieval chain.

        The narrowing steps of Phase 2 are what let a reachability failure
        be localised to a specific component rather than to the pipeline
        as a whole.

        ``stage`` is a label (``kgp:narrowingStage``), not a class: the
        stage at which the correct answer was lost is identified by the
        stage label together with the cardinalities.

        ``gold`` is the known-correct value where one exists. When
        supplied, each stage records whether it survived, so stage
        attribution is a graph query. One boolean per stage costs far less
        than recording every excluded member.
        """
        aid = ids.activity_id(f"narrow_{stage}", *discriminators)
        u = iri(aid)
        g = self.g
        g.add((u, RDF.type, PROV.Activity))
        g.add((u, RDF.type, KGP.NarrowingActivity))
        g.add((u, KGP.narrowingStage, Literal(stage)))
        g.add((u, PROV.startedAtTime, _now()))
        if agent:
            g.add((u, PROV.wasAssociatedWith, iri(agent)))

        in_u = self.record_space(input_space)
        g.add((u, PROV.used, in_u))
        g.add((u, KGP.inputCardinality,
               Literal(len(input_space.members), datatype=XSD.integer)))

        n = _Narrowing(self, u, aid, input_space, gold=gold)
        try:
            yield n
        finally:
            g.add((u, PROV.endedAtTime, _now()))

    @contextmanager
    def attestation(self, *, agent: str, subject: str,
                    discriminators: Sequence[Any] = ()) -> Iterator["_Attestation"]:
        """Wrap a check of an existing claim against an external authority.

        Modelled as an attestation, not a derivation: the identifier was
        checked against the registry, not derived from it. CQ6 relies on
        that distinction.
        """
        aid = ids.activity_id("attest", *discriminators)
        u = iri(aid)
        g = self.g
        g.add((u, RDF.type, PROV.Activity))
        g.add((u, RDF.type, KGP.Attestation))
        g.add((u, PROV.startedAtTime, _now()))
        g.add((u, PROV.wasAssociatedWith, iri(agent)))
        g.add((u, PROV.used, iri(subject)))
        a = _Attestation(self, u, aid, subject)
        try:
            yield a
        finally:
            g.add((u, PROV.endedAtTime, _now()))


    # -- assessment and decision transfer ----------------------------

    def gold_annotation(self, local_id: str, assesses: str, value: str,
                        annotator: str = "corpus_annotator") -> URIRef:
        """Reference annotation from an external corpus.

        Not part of the derivation: an independent assessment of it,
        attributed to a human agent. CQ2 reads this to separate
        reachability failure from selection failure.
        """
        u = iri(local_id)
        g = self.g
        g.add((u, RDF.type, PROV.Entity))
        g.add((u, RDF.type, KGP.GoldAnnotation))
        g.add((u, KGP.goldValue, Literal(value)))
        g.add((u, KGP.assesses, iri(assesses)))
        g.add((u, PROV.wasInfluencedBy, iri(assesses)))
        g.add((u, PROV.wasAttributedTo, iri(annotator)))
        g.add((iri(annotator), RDF.type, PROV.Person))
        return u

    def decision_transfer(self, receiver: str, decided_from: str,
                          dedup_key: str) -> URIRef:
        """Record that an entity received a determination made elsewhere.

        Deduplication decides one mention's identity in another
        mention's context. The deciding activity therefore generates
        entities it never used, which is structurally anomalous and
        invisible unless recorded. CQ8 reads this.
        """
        g = self.g
        ku = iri(dedup_key)
        g.add((ku, RDF.type, KGP.DeduplicationKey))
        dt_node = BNode()
        g.add((iri(receiver), KGP.decisionTransferredFrom, iri(decided_from)))
        g.add((iri(receiver), PROV.qualifiedInfluence, dt_node))
        g.add((dt_node, RDF.type, KGP.DecisionTransfer))
        g.add((dt_node, KGP.decidingContext, iri(decided_from)))
        g.add((dt_node, PROV.entity, ku))
        return dt_node

    def ontology_snapshot(self, local_id: str, version: str,
                          revision_of: str | None = None) -> URIRef:
        u = iri(local_id)
        self.g.add((u, RDF.type, KGP.OntologySnapshot))
        self.g.add((u, RDF.type, PROV.Entity))
        self.g.add((u, KGP.ontologyVersion, Literal(version)))
        self.g.add((u, PROV.generatedAtTime, _now()))
        if revision_of:
            self.g.add((u, PROV.wasRevisionOf, iri(revision_of)))
        return u

    # -- output ------------------------------------------------------


    def serialize(self, path: str, fmt: str = "turtle") -> None:
        self.g.serialize(destination=path, format=fmt)

    def __len__(self) -> int:
        return len(self.g)


# ---------------------------------------------------------------------
#  Handles yielded by the context managers
# ---------------------------------------------------------------------

class _Call:
    def __init__(self, rec: Recorder, uri: URIRef, aid: str):
        self.rec, self.uri, self.aid = rec, uri, aid

    def constrained_by(self, cs: ConstraintSpace, *, enforcement: str,
                       enforcement_artifact: str | None = None) -> None:
        """Bind a space to this call, recording *how* it was enforced.

        ``enforcement`` is required rather than defaulted, so every call
        site states how the space was applied. A default of ``DECODE_TIME``
        would record decode-time masking for a call site that only
        validates afterwards.

        ``enforcement_artifact`` is a content hash of the schema actually
        sent, not the schema itself. The admissible values are already in
        the space's qualified memberships, and Phase 2 compiles a fresh
        candidate enum per call, so storing the JSON again would duplicate
        the largest part of the graph. The hash allows checking that the
        schema sent matches the space recorded.
        """
        if enforcement not in ENFORCEMENT_MODES:
            raise ValueError(
                f"enforcement must be one of {sorted(ENFORCEMENT_MODES)}, "
                f"got {enforcement!r}")
        su = self.rec.record_space(cs)
        self.rec.g.add((self.uri, KGP.wasConstrainedBy, su))
        self.rec.g.add((self.uri, PROV.used, su))
        self.rec.g.add((su, KGP.constraintEnforcement, Literal(enforcement)))
        if enforcement_artifact:
            self.rec.g.add((su, KGP.enforcementArtifact,
                            Literal(enforcement_artifact)))

    def used(self, local_id: str) -> None:
        self.rec.g.add((self.uri, PROV.used, iri(local_id)))

    def generated(self, local_id: str, cls: URIRef | None = None, **attrs) -> URIRef:
        u = self.rec.entity(local_id, cls=cls, **attrs)
        self.rec.g.add((u, PROV.wasGeneratedBy, self.uri))
        return u

    def rationale(self, text: str) -> URIRef:
        """Record a model-produced reasoning trace.

        Attached as a co-generated Entity, never as a description of the
        activity: a rationale is output, not an observed trace, and its
        faithfulness to the underlying computation is not established.
        It must not appear in a derivation chain.
        """
        u = iri(f"{self.aid}_rationale")
        self.rec.g.add((u, RDF.type, PROV.Entity))
        self.rec.g.add((u, RDF.type, KGP.GeneratedRationale))
        self.rec.g.add((u, PROV.wasGeneratedBy, self.uri))
        self.rec.g.add((u, KGP.rationaleText, Literal(text)))
        return u

    def not_repeatable(self) -> None:
        self.rec.g.add((self.uri, KGP.isRepeatable,
                        Literal(False, datatype=XSD.boolean)))


class _Narrowing:
    def __init__(self, rec: Recorder, uri: URIRef, aid: str,
                 in_space: ConstraintSpace, gold: str | None = None):
        self.rec, self.uri, self.aid, self.in_space = rec, uri, aid, in_space
        self.gold = gold

    def excluded(self, values: Iterable[str], reason: str) -> None:
        g = self.rec.g
        g.add((self.uri, KGP.exclusionReason, Literal(reason)))
        for v in values:
            vu = iri(f"{self.in_space.space_id}_m_{ids.content_hash(str(v))[:8]}")
            g.add((self.uri, KGP.excluded, vu))

    def output_space(self, cs: ConstraintSpace,
                     true_members: Iterable[str] | None = None
                     ) -> ConstraintSpace:
        """Record the reduced space.

        ``true_members`` supplies the real membership when ``cs`` is a
        compacted stand-in. Compaction writes empty-member spaces to keep
        the graph small, so ``cs.contains(gold)`` would be uniformly False
        and the gold witness would report a loss at every stage. The
        witness is computed against what the stage actually produced.
        """
        # Materialize once: this is read twice below, and a generator would
        # be exhausted after the first read.
        true_list = None if true_members is None else list(true_members)
        cs.narrowed_from = self.in_space.space_id
        out_u = self.rec.record_space(cs)
        g = self.rec.g
        g.add((out_u, PROV.wasGeneratedBy, self.uri))
        g.add((self.uri, KGP.outputCardinality,
               Literal(len(cs.members), datatype=XSD.integer)))

        # Overwrite the space's own kgp:cardinality as well. `record_space`
        # counts the member list, which compaction empties, so a compacted
        # space would otherwise report 0. CQ3 reads this value as the output
        # size of each narrowing stage.
        if true_list is not None:
            g.remove((out_u, KGP.cardinality, None))
            g.add((out_u, KGP.cardinality,
                   Literal(len(true_list), datatype=XSD.integer)))

        # Did the known-correct value survive this reduction? One boolean
        # makes stage attribution answerable in-graph. Recorded only when
        # gold is known, so production traces without ground truth carry
        # no meaningless witness.
        if self.gold is not None:
            present = (self.gold in set(true_list)
                       if true_list is not None
                       else cs.contains(self.gold))
            g.add((self.uri, KGP.goldSurvived,
                   Literal(present, datatype=XSD.boolean)))
        return cs


class _Attestation:
    def __init__(self, rec: Recorder, uri: URIRef, aid: str, subject: str):
        self.rec, self.uri, self.aid, self.subject = rec, uri, aid, subject

    def against(self, source_iri: str, retrieved_at: str | None = None,
                specialization_of: str | None = None) -> URIRef:
        """Register the authority state consulted.

        Modelled as a time-indexed specialization of a general resource,
        never as a mutable collection: "the web" is not an entity whose
        membership can be asserted.
        """
        u = iri(f"snapshot_{ids.content_hash(source_iri)[:12]}")
        g = self.rec.g
        g.add((u, RDF.type, PROV.Entity))
        g.add((u, KGP.sourceUrl, Literal(source_iri)))
        g.add((u, PROV.generatedAtTime, Literal(retrieved_at) if retrieved_at else _now()))
        if specialization_of:
            g.add((u, PROV.specializationOf, iri(specialization_of)))
        g.add((self.uri, KGP.attestedAgainst, u))
        g.add((self.uri, PROV.used, u))
        return u

    def outcome(self, outcome: URIRef, repeatable: bool = False) -> None:
        self.rec.g.add((self.uri, KGP.attestationOutcome, outcome))
        self.rec.g.add((self.uri, KGP.isRepeatable,
                        Literal(repeatable, datatype=XSD.boolean)))
