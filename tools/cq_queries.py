#!/usr/bin/env python3
"""
The competency questions, as executable SPARQL.

Each query is run against a captured provenance graph. A query that
returns rows is a question the model can answer; a query that returns
nothing points at an unpopulated graph or an inexpressible question.

The bucket on each query records the same distinction as the ontology's
kgp:bucket annotation: A means the question rests on a core PROV relation,
B means it needs a profile term. Bucket A queries still use kgp: classes
to select their nodes.

Run:
    python tools/cq_queries.py out/provenance/stage1_weak_clean.ttl
    python tools/cq_queries.py out/provenance/stage1_weak_clean.ttl --cq CQ3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rdflib import Graph

PREFIXES = """
PREFIX kgp:  <https://w3id.org/kgprov#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


# =====================================================================
#  Each CQ: natural-language question, bucket, SPARQL, and a one-line
#  reading of what a non-empty result demonstrates.
# =====================================================================

CQS: dict[str, dict] = {}


def cq(cid, question, bucket, sparql, demonstrates, agg=None,
       limitation=None):
    """Register one competency question.

    ``limitation`` names a respect in which the query falls short of the
    natural-language question against the graphs this pipeline currently
    emits. It is printed with the result so that a query returning rows
    is never mistaken for a question fully answered.
    """
    CQS[cid] = dict(question=question, bucket=bucket, sparql=sparql,
                    demonstrates=demonstrates, agg=agg,
                    limitation=limitation)


def terms_named(sparql: str) -> set[str]:
    """Local names of every kgp: term a query's text mentions.

    Both syntaxes, since a query may use the prefix or the full IRI. This
    is the operative reading of the pruning criterion: a term survives if
    a question's SPARQL names it, so what counts as "names it" belongs
    next to the queries rather than in each caller.
    """
    return (set(re.findall(r"kgp:([A-Za-z][A-Za-z0-9_]*)", sparql))
            | set(re.findall(
                r"<https://w3id\.org/kgprov#([A-Za-z][A-Za-z0-9_]*)>",
                sparql)))


def _optional_spans(sparql: str) -> list[tuple[int, int]]:
    """Character ranges covered by each OPTIONAL block, braces balanced."""
    spans = []
    for m in re.finditer(r"OPTIONAL\s*\{", sparql):
        depth, i = 0, m.end() - 1
        while i < len(sparql):
            if sparql[i] == "{":
                depth += 1
            elif sparql[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((m.start(), i))
    return spans


def terms_named_optionally(sparql: str) -> set[str]:
    """Terms this query names only inside an OPTIONAL.

    Deleting one costs the query a column; it does not stop the query
    returning rows. Worth reporting separately so "the query traverses
    it" is not read as "the query depends on it".
    """
    spans = _optional_spans(sparql)
    out = set()
    for term in terms_named(sparql):
        hits = list(re.finditer(r"kgp:" + re.escape(term) + r"\b", sparql))
        if hits and all(any(a <= h.start() <= b for a, b in spans)
                        for h in hits):
            out.add(term)
    return out


def parked_cq(*args, **kwargs):
    """Same signature as cq(), but registers nothing.

    A query passed to parked_cq stays in the source and is left out of
    CQS, so neither runner executes it. No query in this module uses it.
    """
    return None


# ---- Family 1: constraint and reachability --------------------------

cq("CQ1",
   "For a given edge, what alternatives were available at the phase that produced it, and were they static or derived?",
   "B",
   # The question is asked of an edge, so the query starts at one and
   # reaches the space through the activity that generated the edge.
   # Only the Phase 3 space bounds an edge.
   #
   # ?card is the recorded cardinality, read once per space. Counting
   # prov:hadMember here instead would multiply the row by the membership
   # and bias the mean toward the larger spaces.
   """
   SELECT ?phase ?kind ?space_provenance
          (COUNT(DISTINCT ?edge) AS ?edges)
          (AVG(?card) AS ?mean_alternatives)
   WHERE {
     ?edge a kgp:KnowledgeGraphEdge ; prov:wasGeneratedBy ?act .
     ?act kgp:phase ?phase ; kgp:wasConstrainedBy ?space .
     ?space kgp:spaceKind ?kind ; kgp:cardinality ?card .
     OPTIONAL { ?space a kgp:DerivedConstraintSpace . BIND("derived" AS ?d) }
     OPTIONAL { ?space a kgp:StaticConstraintSpace  . BIND("static"  AS ?s) }
     BIND(COALESCE(?d, ?s, "unclassified") AS ?space_provenance)
   } GROUP BY ?phase ?kind ?space_provenance ORDER BY DESC(?edges)
   """,
   "the alternatives an edge was drawn from are reachable from the edge itself, and each space says whether it was recoverable from the plan or computed during the run")

cq("CQ2",
   "Across grounded mentions with a gold identifier, what share of errors are reachability vs selection failures?",
   "B",
   # The gold annotation is joined to the mention it assesses, and the
   # mention to the constraint space that bounded its own grounding call,
   # so a candidate counts as gold only for the mention it is gold for.
   #
   # Use tools/cq_queries_oxi.py, not the rdflib runner. An EXISTS-based
   # form is slower than the OPTIONAL form below.
   #
   # Where the gold value was in the space and lost, the rank and score
   # columns say how close it came. Both are read from the gold member's
   # own qualified membership, so an unscored membership leaves the
   # column empty instead of dropping the row.
   """
   SELECT ?outcome ?kind
          (COUNT(DISTINCT ?pair) AS ?n)
          (AVG(?card) AS ?mean_space_size)
          (AVG(?grank) AS ?mean_gold_rank)
          (AVG(?gscore) AS ?mean_gold_score)
   WHERE {
     ?act kgp:wasConstrainedBy ?space ;
          prov:used ?ent ;
          prov:wasAssociatedWith ?agent .
     ?agent a kgp:PrimaryEvaluatorAgent .
     ?space kgp:spaceKind ?kind ; kgp:cardinality ?card .
     ?ent a kgp:GroundedMention .
     ?gold a kgp:GoldAnnotation ; kgp:assesses ?ent ; kgp:goldValue ?gv .
     BIND(CONCAT(STR(?act), "|", STR(?ent)) AS ?pair)

     # Is the gold value a member of THIS mention's space?
     OPTIONAL {
       ?space kgp:qualifiedMembership ?gqm .
       ?gqm a kgp:SpaceMembership ; kgp:member ?gmem ; kgp:rank ?grank .
       ?gmem kgp:memberValue ?gv .
       OPTIONAL { ?gqm kgp:score ?gscore }
     }
     # Which member did this activity actually select?
     OPTIONAL {
       ?space kgp:qualifiedMembership ?sqm .
       ?sqm kgp:wasSelected true ; kgp:member ?smem .
       ?smem kgp:memberValue ?sv .
     }
     BIND(IF(!BOUND(?gmem), "UNREACHABLE",
          IF(BOUND(?sv) && ?sv = ?gv, "REACHABLE_SELECTED",
             "REACHABLE_NOT_SELECTED")) AS ?outcome)
   } GROUP BY ?outcome ?kind ORDER BY DESC(?n)
   """,
   "reachability failure is distinguishable from selection failure per mention",
   agg="reachability",
   limitation=("the query counts (grounding call, mention) pairs in the "
               "graph, which is not the population of mentions, for two "
               "reasons. "
               "(1) phases.py records a constrained generation only when "
               "the candidate pool is non-empty, so keys whose pool the "
               "score floor suppressed -- the single largest class of "
               "reachability failure -- are absent entirely, biasing this "
               "split toward selection. (2) Only the first mention per "
               "dedup key in each document is linked via prov:used, and "
               "every Tier 1 arm is counted, so the counts are per (key, "
               "arm, document), not per mention. Mention-level figures "
               "come from out/results/reachability_<split>.csv "
               "(pipeline/retrieval_eval.py) and tools/exact_selection.py. "
               "Recording suppressed pools as empty constraint spaces "
               "would close gap (1)."))

cq("CQ3",
   "For an unreachable gold identifier, at which narrowing step did it disappear?",
   "B",
   # Stages are identified by kgp:narrowingStage, not by subclass: the
   # question is answered by the stage identity plus the cardinalities.
   #
   # kgp:goldSurvived records, per stage, whether the gold value was still
   # in the output space, which makes the question answerable in-graph.
   # Each stage generates the space it produced, and that space records
   # its predecessor through kgp:narrowedFrom and its own size, so the
   # reduction is readable from the graph. kgp:excluded and
   # kgp:exclusionReason say what left and why.
   #
   # Each of the three optional properties is at most one per activity,
   # so the row is not multiplied and SUM stays a sum over activities.
   # The query is written flat rather than as grouped subqueries joined
   # on ?stage.
   """
   SELECT ?stage
          (COUNT(DISTINCT ?act) AS ?invocations)
          (SUM(IF(?survived, 0, 1)) AS ?gold_absent_after)
          (SUM(?in - ?out) AS ?candidates_removed)
          (SUM(IF(?out = 0, 1, 0)) AS ?spaces_emptied)
          (AVG(?card) AS ?mean_output_size)
          (COUNT(?w) AS ?exclusions_recorded)
          (GROUP_CONCAT(DISTINCT ?why; separator=", ") AS ?exclusion_reason)
   WHERE {
     ?act a kgp:NarrowingActivity ;
          kgp:narrowingStage ?stage ;
          kgp:inputCardinality ?in ; kgp:outputCardinality ?out .
     ?outspace a kgp:DerivedConstraintSpace ;
               prov:wasGeneratedBy ?act ;
               kgp:narrowedFrom ?prevspace ;
               kgp:cardinality ?card .
     OPTIONAL { ?act kgp:goldSurvived ?survived }
     OPTIONAL { ?act kgp:excluded ?excluded }
     OPTIONAL { ?act kgp:exclusionReason ?w }
     BIND(COALESCE(?w, "(none recorded)") AS ?why)
   } GROUP BY ?stage ORDER BY DESC(?candidates_removed)
   """,
   "the stage at which a known-correct answer left the space is recorded per stage, so a constraint space has internal provenance and the loss is attributable to one component",
   agg="narrowing",
   limitation=("kgp:goldSurvived is written only to the reachability "
               "witness graph that python -m pipeline.retrieval_eval "
               "--emit-witness writes, so against the main release graph "
               "the gold_absent_after column is empty and only the stage "
               "cardinalities are returned; run this against the "
               "witness graph for the attribution. "
               "kgp:excluded is likewise degraded in the compact graph: "
               "all 4,885 of its uses point at one placeholder node, so "
               "exclusions_recorded counts the activities that recorded "
               "an exclusion while the excluded members themselves are "
               "named only in the full-narrowing graph, where there are "
               "510,076 of them. Beyond "
               "that, gold_absent_after counts every stage at or after the "
               "loss, because a value missing from one output is missing "
               "from all downstream ones. The loss point is the first such "
               "stage in chain order (DENSE_RETRIEVAL, SPARSE_RETRIEVAL, "
               "RANK_FUSION, TYPE_FILTER, CROSS_ENCODER_RERANK, "
               "SCORE_FLOOR); the query reports per-stage counts "
               "and leaves that ordering to the reader. Recording a stage "
               "ordinal would let the query take the minimum directly. "
               "One row carries a known capture defect. "
               "CROSS_ENCODER_RERANK reports the same mean_output_size as "
               "TYPE_FILTER, because the space it generates is stamped "
               "with the size of the pool it scored rather than the size "
               "of the top-k it kept, so the reranker is the one link in "
               "the chain where the space disagrees with the activity "
               "that produced it, and with the inputCardinality of the "
               "score floor that consumed it. The same defect makes "
               "kgp:goldSurvived at this stage mean 'was scored' rather "
               "than 'survived the cut', which is why the witness graph "
               "reports the same true/false split here as for "
               "TYPE_FILTER: in-graph attribution is blind at this stage. "
               "The activity's own inputCardinality and outputCardinality "
               "are correct and candidates_removed is computed from them. "
               "The per-stage columns of "
               "out/results/reachability_<split>.csv are computed by "
               "pipeline/retrieval_eval.py from the retrieval result, not "
               "from the recorded spaces, so they do not depend on the "
               "defect. The fix is to truncate to the cut before "
               "recording; the code is left unchanged so that it "
               "reproduces the released graph."))

cq("CQ4",
   "Do the same phases run under different configuration variants, enabling counterfactual comparison?",
   "B",
   # The counterfactual is a join of the same content-addressed mention
   # across two or more run bundles. This query does not perform that
   # join. It counts what identifies each run for the join: the variant
   # label, the pinned model and the code version. The join itself needs
   # at least two bundles in the graph.
   """
   SELECT (COUNT(DISTINCT ?run) AS ?runs)
          (COUNT(DISTINCT ?variant) AS ?configuration_variants)
          (COUNT(DISTINCT ?model) AS ?models)
          (COUNT(DISTINCT ?code) AS ?pipeline_versions)
   WHERE {
     ?run a kgp:PipelineRun ;
          kgp:configurationVariantLabel ?variant ;
          kgp:modelId ?model ;
          kgp:pipelineVersion ?code .
   }
   """,
   "content-addressed mention identifiers make cross-configuration comparison a join over run bundles rather than a re-execution",
   limitation=("requires >= 2 run bundles in one graph. On a single-run "
               "graph the query returns one row with runs = 1, so the "
               "cross-run comparison is not exercised."))

# ---- Family 2: attestation and revision -----------------------------

cq("CQ5",
   "Which identifiers were retired or merged between MeSH releases, and by which provenance path were they produced?",
   "A",
   # The question needs the ontology snapshot, its release identifier,
   # and the core PROV revision edge between snapshots.
   """
   SELECT ?release
          (COUNT(DISTINCT ?snap) AS ?snapshots)
          (COUNT(DISTINCT ?revised) AS ?snapshots_revised)
   WHERE {
     ?snap a kgp:OntologySnapshot ; kgp:ontologyVersion ?release .
     OPTIONAL { ?revised prov:wasRevisionOf ?snap }
   } GROUP BY ?release
   """,
   "ontology drift is expressible in core PROV via wasRevisionOf against time-indexed snapshots, and the release each snapshot stands for is recorded",
   limitation=("snapshots_revised is 0. Both MeSH releases are recorded "
               "as snapshots, but the pipeline never asserts a revision "
               "edge between them: prov:wasRevisionOf occurs zero times "
               "in the graph. The question is expressible and the "
               "vocabulary is present; the run does not exercise it."))

cq("CQ6",
   "Which identifiers were attested against external state that is not repeatable, and what was the outcome?",
   "B",
   # kgp:attestedAgainst names the authority state each claim was checked
   # against, which is what makes the check unrepeatable. The query groups
   # by repeatability instead of filtering on it, so the result shows the
   # repeatable attestations as well as the ones that are not.
   """
   SELECT ?outcome ?repeatable
          (COUNT(DISTINCT ?a) AS ?attestations)
          (COUNT(DISTINCT ?authority) AS ?authority_states)
          (COUNT(DISTINCT ?url) AS ?distinct_sources)
   WHERE {
     ?a a kgp:Attestation ;
        kgp:attestationOutcome ?outcome ;
        kgp:isRepeatable ?repeatable ;
        kgp:attestedAgainst ?authority .
     OPTIONAL { ?authority kgp:sourceUrl ?url }
   } GROUP BY ?outcome ?repeatable ORDER BY DESC(?attestations)
   """,
   "attestation is modelled distinctly from derivation, the authority state each claim was checked against is recorded, and irrepeatability is a property of the check rather than an assumption about it",
   agg="attestation",
   limitation=("the authority states are typed prov:Entity and carry "
               "kgp:sourceUrl, but none is typed kgp:OntologySnapshot, so "
               "the release an attestation was made against cannot be "
               "reached from the attestation. Typing the snapshot at "
               "capture time would close it. Note also that the 495 "
               "Unverifiable checks share a single authority state, the "
               "not-found sentinel."))

cq("CQ7",
   "For provisional identifiers later revised, does the revision link survive to support before/after comparison?",
   "A",
   # The question is about revision links, so the query looks for
   # prov:wasRevisionOf as well as counting provisional nodes. Reporting
   # both puts the revision count next to the number of nodes that could
   # have been revised.
   """
   SELECT (COUNT(DISTINCT ?prov_id) AS ?provisional_nodes)
          (COUNT(DISTINCT ?revised) AS ?provisional_nodes_revised)
   WHERE {
     ?prov_id a kgp:ProvisionalConceptID .
     OPTIONAL {
       ?newer prov:wasRevisionOf ?prov_id .
       BIND(?prov_id AS ?revised)
     }
   }
   """,
   "provisional nodes are first-class and content-addressed, so a later merge would be a wasRevisionOf edge rather than a new node",
   limitation=("provisional_nodes_revised is 0 on this run: the pipeline "
               "has no reconciliation pass, so the event the question asks "
               "about never occurs. The question is expressible and the "
               "machinery instrumented; the corpus does not exercise it."))

# ---- Family 3: contextual displacement ------------------------------

cq("CQ8",
   "Which mentions were assigned an identity decided in a context that did not include them?",
   "B",
   # Counts the recorded decision transfers and the mentions that
   # received them. How the receiving context differed is not recorded.
   """
   SELECT (COUNT(?dt) AS ?transfers) (COUNT(DISTINCT ?receiver) AS ?mentions)
   WHERE {
     ?receiver prov:qualifiedInfluence ?dt .
     ?dt a kgp:DecisionTransfer ; kgp:decidingContext ?ctx .
   }
   """,
   "deduplication as decision transfer is recorded: the receiving mention, and the context the determination was made in",
   agg="displacement",
   limitation=("the transfer is recorded but not characterised. Whether the "
               "receiving context materially differed from the deciding one "
               "is not computed, and Phase 2 could not compute all of it "
               "anyway, since polarity and co-occurrence are Phase 3 "
               "properties."))

# ---- Family 4: negative provenance ----------------------------------

cq("CQ9",
   "What was extracted but never reached the graph, and under which reason code?",
   "A",
   # Each rejection is reported with the type of what was lost (edge,
   # action phrase or raw span), since the review queue is sorted by that
   # distinction. These three are the types the pipeline invalidates.
   """
   SELECT ?reason ?extracted (COUNT(DISTINCT ?inv) AS ?n) WHERE {
     ?e prov:qualifiedInvalidation ?inv .
     ?inv kgp:hadRejectionReason ?reason .
     ?reason a kgp:RejectionReason .
     OPTIONAL { ?e a kgp:KnowledgeGraphEdge . BIND("edge" AS ?k1) }
     OPTIONAL { ?e a kgp:VerbatimAction    . BIND("action phrase" AS ?k2) }
     OPTIONAL { ?e a kgp:RawSpan           . BIND("raw span" AS ?k3) }
     BIND(COALESCE(?k1, ?k2, ?k3, "untyped") AS ?extracted)
   } GROUP BY ?reason ?extracted ORDER BY DESC(?n)
   """,
   "the rejection reasons make the review queue sortable by failure mode, and each rejection says what kind of thing was lost; six of the eight declared codes fire, and GROUP BY cannot show the two that do not",
   agg="rejection",
   limitation=("the reason codes are individuals of kgp:RejectionReason, "
               "and that typing lives in the vocabulary rather than in "
               "the run graph, so a query over the data alone cannot ask "
               "for it. Only 125 of the 880 invalidations carry "
               "prov:activity, so the phase a rejection happened in is "
               "not reachable for most of them either."))

cq("CQ10",
   "Can 'the source denies this' (polarity) be distinguished from 'we are unsure' (modality/confidence)?",
   "B",
   """
   SELECT ?polarity ?modality (COUNT(?edge) AS ?n) WHERE {
     ?edge a kgp:KnowledgeGraphEdge ; kgp:polarity ?polarity ;
           kgp:modality ?modality .
   } GROUP BY ?polarity ?modality ORDER BY DESC(?n)
   """,
   "source-level negation and record-level uncertainty are orthogonal axes, both queryable and not conflated")


# ---- Enforcement and unbounded generation (CQ11-CQ13) ----------------
#
# Registered with cq() like CQ1-CQ10, so they are in CQS and both runners
# execute them. Which questions count as active for pruning is set by
# DEFAULT_ACTIVE in tools/prune_profile.py, not here.

cq("CQ11",
   "Was the constraint enforced at decode time or checked afterwards?",
   "B",
   # Enforcement is a property of the space, not of the activity: the
   # space carries the mode and the hash of the artefact that imposed it.
   #
   # Grouped by phase as well, because that is where the answer varies:
   # the Phase 1 and Phase 4 enums are recorded as static and the Phase 2
   # and Phase 3 spaces as derived. The prompt hash is carried because a
   # POST_VALIDATED constraint is one stated in the prompt and checked
   # afterwards, so the instruction that carried it is part of the answer.
   """
   SELECT ?mode ?space_provenance ?phase
          (COUNT(DISTINCT ?space) AS ?spaces)
          (COUNT(DISTINCT ?act) AS ?activities)
          (COUNT(DISTINCT ?artifact) AS ?artifacts)
          (COUNT(DISTINCT ?prompt_hash) AS ?prompt_hashes)
          (COUNT(DISTINCT ?mention) AS ?values_confirmed_in_space)
   WHERE {
     ?space a kgp:ConstraintSpace ;
            kgp:constraintEnforcement ?mode ;
            kgp:enforcementArtifact ?artifact .
     ?act kgp:wasConstrainedBy ?space ; kgp:phase ?phase .
     OPTIONAL { ?space a kgp:StaticConstraintSpace  . BIND("static"  AS ?s) }
     OPTIONAL { ?space a kgp:DerivedConstraintSpace . BIND("derived" AS ?d) }
     BIND(COALESCE(?s, ?d, "unclassified") AS ?space_provenance)
     OPTIONAL {
       ?act prov:used ?pt .
       ?pt a kgp:PromptTemplate ; kgp:promptHash ?prompt_hash .
     }
     # Check the enforcement claim rather than take it on trust: under
     # DECODE_TIME every produced value must be a member of the space
     # that bounded it. Only Phase 1 exposes the produced value as an
     # attribute of the entity it generated.
     OPTIONAL {
       ?space prov:hadMember ?allowed_member .
       ?allowed_member kgp:memberValue ?allowed .
       ?mention prov:wasGeneratedBy ?act ; kgp:semanticCategory ?produced .
       FILTER(UCASE(?produced) = UCASE(?allowed))
     }
   } GROUP BY ?mode ?space_provenance ?phase ORDER BY DESC(?spaces)
   """,
   "the enforcement regime is a recorded property of each space, backed by the hash of the artefact that imposed it, so a constraint described in a prompt is distinguishable from one compiled into a schema",
   limitation=("every space in this run is DECODE_TIME, so the query "
               "separates the two regimes without observing both. A "
               "prompt-level (POST_VALIDATED) arm would populate the "
               "other row; no call site in this pipeline records one. The "
               "confirmation column compares case-insensitively: the "
               "Phase 1 enum records CHEMICAL and DISEASE while the "
               "mentions it bounded record Chemical and Disease, so an "
               "exact match reports zero. The mismatch is in the record "
               "rather than in the generation, and it is the reason the "
               "enforcement claim needs checking rather than trusting."))

cq("CQ12",
   "Where no constraint space exists, what carries the graph, and how much rests on it?",
   "B",
   # Three independent counts in one query rather than a join, because
   # the join does not exist: kgp:UnboundedGeneration carries a budget
   # and a termination reason and nothing else. It has no prov:used or
   # prov:generated edge to the identifier the escalation produced, so
   # the two halves can only be related through the mappingTier value.
   """
   SELECT ?dimension ?value (COUNT(DISTINCT ?x) AS ?n) WHERE {
     {
       ?x a kgp:CanonicalConceptID ; kgp:mappingTier ?value ;
          kgp:cuiValue ?cui .
       BIND("identifier by mapping tier" AS ?dimension)
     } UNION {
       ?x a kgp:ProvisionalConceptID .
       BIND("identifier is provisional" AS ?dimension)
       BIND("yes" AS ?value)
     } UNION {
       ?x a kgp:UnboundedGeneration ; kgp:terminationReason ?value .
       BIND("unbounded step by termination" AS ?dimension)
     }
   } GROUP BY ?dimension ?value ORDER BY ?dimension DESC(?n)
   """,
   "the part of the graph resting on unbounded generation is counted rather than inferred: the mapping tier on each identifier records which route produced it, and the provisional ones are typed",
   agg="provisional",
   limitation=("the unbounded step is not linked to the identifier it "
               "produced. kgp:UnboundedGeneration has no prov:generated "
               "edge, so 'how much rests on it' is answered through the "
               "mappingTier value rather than by traversal. That the WEB "
               "tier and the escalation count agree at 1,130 is "
               "consistent with the intended reading but is not asserted "
               "in the graph; emitting prov:generated would close it."))

cq("CQ13",
   "Does the unbounded agent return identifiers that do not exist, and were failures its own conclusion or the controller's cutoff?",
   "B",
   # Six independent dimensions, joined by UNION rather than by OPTIONAL.
   # The graph has no edge between an attestation and the escalation that
   # produced it, so nesting the escalation and its tool calls in an
   # OPTIONAL is a cross product (971 x 1,130 x 4,601) that does not
   # terminate. The dimensions are counted side by side instead.
   #
   # Two edges do exist and are traversed: the attestation used the
   # identifier it checked, and each tool invocation was informed by the
   # escalation that issued it. The agent is named as its own dimension
   # because "the unbounded agent" in the question is a specific one.
   """
   SELECT ?dimension ?value (COUNT(DISTINCT ?x) AS ?n) WHERE {
     {
       ?x a kgp:Attestation ; kgp:attestationOutcome ?value .
       ?value a kgp:AttestationOutcome .
       BIND("attestation outcome" AS ?dimension)
     } UNION {
       ?x a kgp:Attestation ; prov:used ?cid .
       ?cid a kgp:CanonicalConceptID .
       BIND("identifier checked" AS ?dimension)
       BIND("CanonicalConceptID" AS ?value)
     } UNION {
       ?x a kgp:Attestation ; prov:wasAssociatedWith ?agent .
       ?agent a kgp:EscalationSearchAgent .
       BIND("carried out by" AS ?dimension)
       BIND("EscalationSearchAgent" AS ?value)
     } UNION {
       ?x a kgp:UnboundedGeneration ; kgp:terminationReason ?value ;
          kgp:callBudget ?budget .
       BIND("escalation termination" AS ?dimension)
     } UNION {
       ?x a kgp:ToolInvocation ; kgp:toolName ?value ;
          kgp:toolArguments ?args ; kgp:invocationOrdinal ?ordinal ;
          prov:wasInformedBy ?escalation .
       ?escalation a kgp:UnboundedGeneration .
       BIND("tool call" AS ?dimension)
     } UNION {
       ?x a kgp:VerificationCheck ; kgp:checkKind ?kind ;
          kgp:checkOutcome ?passed .
       BIND(CONCAT(?kind, " passed=", STR(?passed)) AS ?value)
       BIND("verification check" AS ?dimension)
     }
   } GROUP BY ?dimension ?value ORDER BY ?dimension DESC(?n)
   """,
   "fabrication and exhaustion are separable: an identifier that failed checking is a different finding from a search the controller cut off, and the two have different remedies")


# =====================================================================
#  Runner
# =====================================================================

def run_one(g: Graph, cid: str, verbose: bool = True) -> list:
    spec = CQS[cid]
    rows = list(g.query(PREFIXES + spec["sparql"]))
    if verbose:
        bucket = spec["bucket"]
        tag = {"A": "core PROV", "B": "extension", "C": "out of scope"}[bucket]
        print(f"\n{'='*70}")
        print(f"{cid}  [{bucket} - {tag}]")
        print(f"  Q: {spec['question']}")
        print(f"{'-'*70}")
        if not rows:
            print("  (no rows - unpopulated or inexpressible)")
        else:
            cols = rows[0].labels if hasattr(rows[0], "labels") else None
            if cols:
                header = "  " + "  ".join(f"{str(c):>24}" for c in cols)
                print(header)
            for r in rows:
                print("  " + "  ".join(f"{str(v):>24}" for v in r))
        print(f"  -> {spec['demonstrates']}")
        if spec.get("limitation"):
            import textwrap
            print("  !! LIMITATION: " + "\n                 ".join(
                textwrap.wrap(spec["limitation"], 70)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--cq", default=None, help="run a single CQ, e.g. CQ3")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.graph, format="turtle")
    print(f"loaded {len(g):,} triples from {args.graph}")

    targets = [args.cq] if args.cq else list(CQS.keys())
    empty = []
    for cid in targets:
        rows = run_one(g, cid)
        if not rows:
            empty.append(cid)

    limited = [c for c in targets if CQS[c].get("limitation")]

    print(f"\n{'='*70}")
    print(f"ran {len(targets)} queries; "
          f"{len(targets) - len(empty)} returned rows, {len(empty)} empty")
    if empty:
        print(f"EMPTY (investigate): {', '.join(empty)}")
    if limited:
        print(f"ANSWERED WITH A STATED LIMITATION: {', '.join(limited)}")
        print("  A query returning rows is not the same as a question "
              "answered. Read the limitations above before citing these.")
    if not empty and not limited:
        print("all competency questions answerable against this graph")


if __name__ == "__main__":
    main()
