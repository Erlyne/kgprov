# KGPROV

Code, ontology, competency queries and provenance graph for a biomedical
knowledge-graph pipeline whose construction is recorded in W3C PROV.

The pipeline reads the BC5CDR corpus, extracts chemical and disease
mentions and the relations between them, grounds the mentions against
MeSH using constrained language-model calls (AWS Bedrock), and records
the run as a PROV graph. For each constrained call the graph holds the
set of values the model was allowed to produce and how that set was
enforced. For the Phase 2 grounding calls it also marks the candidate the
model selected, and each retrieval stage that narrowed MeSH to the
candidate list is recorded as a narrowing activity with its
cardinalities. The vocabulary is the `kgp:` namespace,
`https://w3id.org/kgprov#`.

All commands in this file are run from the repository root.

## Layout

| Path | Contents |
|---|---|
| `kgprov/` | Capture layer (`capture.py`, `ids.py`) and the Bedrock client (`llm.py`) |
| `kgprov/ontology/` | The vocabulary, its derived profile and modules, and its SHACL shapes |
| `pipeline/` | Loaders, chunking, retrieval and its evaluation, the NCBI registry client, the four pipeline phases, the runner |
| `tools/` | Competency queries, shape validation, ontology pruning and splitting, IRI sanitising, the reachability/selection split (`exact_selection.py`, which needs a run's result tables) |
| `tests/` | Test suite |
| `config/config.yaml` | Run configuration |
| `out/provenance/` | Location of the released graph |

## Ontology

`kgprov/ontology/kgprov.ttl` is the ontology: the maximal, commented model
with 87 terms (37 classes, 11 object properties, 39 datatype properties;
annotation properties and individuals are not counted). The profile and
the two modules are generated from it. The module shapes are generated
from `kgprov-shapes.ttl`.

| File | What it is |
|---|---|
| `kgprov.ttl` | Maximal model. Source of the profile and the modules |
| `kgprov-profile.ttl` | Pruned profile (60 terms), produced by `tools/prune_profile.py`, which also writes `out/results/prune_report.md` |
| `kgprov-core.ttl` | Domain-neutral module (61 terms), produced by `tools/split_modules.py` |
| `kgprov-bio.ttl` | Biomedical module (26 terms), produced by `tools/split_modules.py`; imports the core |
| `kgprov-shapes.ttl` | SHACL shapes for the maximal model; eight of the terms they use are pruned from the profile. No tool here generates it |
| `kgprov-core-shapes.ttl`, `kgprov-bio-shapes.ttl` | Shapes for the two modules, split from `kgprov-shapes.ttl` by `tools/split_modules.py` |

The modules are split from the maximal model, not from the profile.

## Provenance graph

The graph is `out/provenance/stage1_weak_clean.ttl`. It records run
`stage1_weak`: the 500-document BC5CDR development split, Claude Haiku
4.5 for every model call (only the weak arm ran), MeSH 2025 as the
primary release and MeSH 2023 as the prior release, with compact
narrowing. Retrieval is built from MeSH 2025 alone, and Tier 2 can also
accept an identifier confirmed only by the live NCBI registry. MeSH 2023
is parsed and recorded as an ontology snapshot (`mesh_2023`) and is not
used for retrieval; `pipeline/loaders.py` can diff the two releases
(`mesh_diff`) from its command line. The run manifest (not included)
records 553,465 triples for the graph as the run wrote it, before
sanitising. The file has been
passed through `tools/sanitize_ttl.py`, which percent-encodes characters
inside IRIs that strict Turtle parsers such as pyoxigraph reject, for
example the brackets in `[3H]spiperone`.

To validate it against the shapes:

```bash
python tools/validate_shapes.py --graph out/provenance/stage1_weak_clean.ttl
```

## Reading the graph

- `kgp:KnowledgeGraphEdge` nodes (`REL_...`) are Phase 3 relation records. The aggregated
  edges of `kg_table.csv` are not in the graph.
- `kgp:modelId` on the run node lists the model IDs configured for both arms in
  `config/config.yaml` (`weak=...;strong=...`), not the models called. Only the weak arm ran.
- `kgp:mappingTier "WEB"` is written for every primary-arm Tier 2 escalation, including claims
  that end `UNMAPPED` or fail the checks: it marks an escalation, not an identifier found on the
  web. In `mention_table.csv`, `mapping_tier` is `WEB` only for verified Tier 2 identifiers.
- Phase 1 enumerates `CHEMICAL`/`DISEASE`; `kgp:semanticCategory` stores `Chemical`/`Disease`.

## Competency queries

The thirteen queries, CQ1 to CQ13, are defined in `tools/cq_queries.py`.
Run them with the pyoxigraph runner, which loads
`kgprov/ontology/kgprov.ttl` together with the graph:

```bash
python tools/cq_queries_oxi.py out/provenance/stage1_weak_clean.ttl
```

A single query:

```bash
python tools/cq_queries_oxi.py out/provenance/stage1_weak_clean.ttl --cq CQ3
```

Output to a file (also echoed to the terminal):

```bash
python tools/cq_queries_oxi.py out/provenance/stage1_weak_clean.ttl --out cq_report.txt
```

The pyoxigraph runner prints each query's rows but not the `limitation`
notes that `tools/cq_queries.py` attaches to ten of the thirteen queries.
Read those before citing a result.

The `kgp:motivatedByCQ` annotations in the ontology cite query numbers up
to CQ14, while `tools/cq_queries.py` defines CQ1 to CQ13; the default set
used for pruning the profile is `DEFAULT_ACTIVE` in
`tools/prune_profile.py` (CQ2, CQ3, CQ6 and CQ9 to CQ13).

`tools/cq_queries.py` also runs on rdflib alone (positional graph path
and `--cq`; no `--out`) and prints those notes. It does not load the
vocabulary, so CQ9 returns no rows and CQ13 loses its attestation-outcome
rows: the reason codes and outcomes are typed `kgp:RejectionReason` and
`kgp:AttestationOutcome` in `kgprov.ttl`, not in the run graph.
`tools/cq_queries_oxi.py` is the practical runner at this graph size; its
docstring notes that rdflib's SPARQL engine becomes unusable at around
700,000 triples.

## Installation

Python 3.14 was used for development.

```bash
pip install -r requirements.txt
```

On first use the retrieval stack downloads two models from Hugging Face:
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext` and
`cross-encoder/ms-marco-MiniLM-L-6-v2`.

These keys in `config/config.yaml` are not read by the code and are
marked there: `models.weak_arm.name`, `models.weak_arm.temperature`,
`models.strong_arm.name`, `models.strong_arm.temperature`,
`models.strong_arm.max_tokens`, `retrieval.encoder`, `decoding.top_p`,
`corpus.n_docs`, `ablations`, `paths.provenance`, `paths.results` and
`run.concurrency`.

A run writes files that are not meant to be committed and that
`.gitignore` excludes: `out/provenance/<run-id>.ttl`,
`out/provenance/<run-id>_fragments/`, `out/results/`, `out/cq_report/`,
`out/response_log/` and `out/escalation_cache.json`.

## Tests

```bash
python -m pytest tests/ -q
```

Tests that need the BC5CDR development file,
`out/results/reachability_dev.csv` (written by
`pipeline/retrieval_eval.py`), or the thesis and appendix sources
(`paper/thesis.tex`, `out/appendix/app_derivation.tex`, not in this
folder) are skipped when those files are absent. The strict-parser test
in `tests/test_capture.py` is skipped if pyoxigraph is not installed. One
test is marked `xfail`; see Known limitations.

## Rebuilding the graph

A new run can be made. It needs network access, and live Bedrock calls
are billed by AWS. It writes a new graph rather than a copy of the
released one: activities carry wall-clock timestamps
(`kgprov/capture.py`), and model and NCBI responses are fetched live
unless they are already cached.

1. Put `CDR_DevelopmentSet.PubTator.txt` in `data/bc5cdr/`
   (`CDR_TrainingSet.PubTator.txt` and `CDR_TestSet.PubTator.txt` are
   read only for `--split train`, `test` or `all`), and the MeSH
   descriptor releases in `data/mesh/` as `desc2025.xml` and
   `desc2023.xml`.
2. Copy `.env.example` to `.env` in the repository root
   (`pipeline/runner.py` reads it from the working directory) and fill in
   AWS credentials with access to the Claude Haiku 4.5 inference profile.
   The strong arm, which this run does not use, needs Sonnet 4.5. The
   Bedrock region used by the code is `us-east-1`, read from
   `config/config.yaml`. Set `NCBI_EMAIL`, and `NCBI_API_KEY` if you have
   one: the Tier 2 stage searches the NCBI E-utilities MeSH database and
   checks there any identifier that is not in the local MeSH 2025 release.
3. Run:

```bash
python -m pipeline.runner --run-id stage1_weak --split dev --n-docs 500 \
  --tier1-arms weak_arm --tier2-arms weak_arm
```

With the defaults for the other flags (`--primary-arm weak_arm`, no
`--sample-strong`, compact narrowing) this gives the arm, sampling and
MeSH settings recorded in the run manifest. The run writes
`out/provenance/stage1_weak.ttl`, the result tables in
`out/results/stage1_weak/`, and a report from the rdflib query runner in
`out/cq_report/stage1_weak.txt`. `kgprov/capture.py` already
percent-encodes the characters the sanitiser handles. To produce a file
under the released name:

```bash
python tools/sanitize_ttl.py out/provenance/stage1_weak.ttl out/provenance/stage1_weak_clean.ttl
```

The first run embeds the MeSH 2025 surface forms and caches the FAISS
index in `data/mesh/index_2025/`; later runs load it. A run of this size
makes several thousand Bedrock calls and is billed by AWS. Every response
is written to `out/response_log/stage1_weak/` and reused by later runs
with the same run id. `--replay` makes no Bedrock calls. Under
`--replay` a call missing from the log raises an error; the phase prints
an `ERROR` line and the run continues, and the graph does not record
that the response was missing. A Phase 1 chunk then yields no mentions
and an `LLM_ERROR` row in `rejected_spans.csv`; a Tier 1 call is
recorded as `DECLINED`, which triggers escalation; a Tier 2 escalation
not in the escalation cache ends with termination reason `TOOL_FAILURE`;
a Phase 3 window yields no relations; a Phase 4 batch marks its rows
`kgp:DroppedFromBatch`. Check the output for `ERROR` lines. The log is
not included here. Under `--replay` the Tier 2 identifier checks still
query NCBI. Tier 2 results are cached in `out/escalation_cache.json`,
also not included.

## Known limitations

- **Exclusion sets are compacted.** In the default compact mode each
  narrowing activity records its input and output cardinalities but not
  the members it removed. The type filter, reranker and score floor point
  `kgp:excluded` at a single placeholder node
  (`compact_excluded_placeholder`); rank fusion records only an exclusion
  reason; dense and sparse retrieval record neither. The graph therefore
  cannot say which stage removed a particular gold answer. `kgp:excluded`
  is declared in `kgprov.ttl`, and `--full-narrowing` writes the excluded
  members of rank fusion, the type filter, the reranker and the score
  floor; dense and sparse retrieval record no exclusions even then.
- **Grounding spaces are not linked to the retrieval chain.** The
  candidate space a Phase 2 grounding call is constrained by has no
  `kgp:narrowedFrom` or `prov:wasDerivedFrom` edge to the narrowing
  activities that produced its candidates (`pipeline/phases.py`, Tier 1
  block), so the chain is not reachable from the call by traversal.
- **`kgp:probabilityMass` is declared and never populated.**
  `kgprov/capture.py` writes it only when a `SpaceMember` carries a
  `probability_mass`, and no code in this folder sets one. The Bedrock
  calls in `kgprov/llm.py` do not request token probabilities.
- **Reranker output cardinality.** In compact mode the `CROSS_ENCODER_RERANK`
  output space is sized by the scored pool, not the kept top-k
  (`true_members=` in `pipeline/retrieval.py`). The activity's cardinalities
  are correct; `kgp:goldSurvived` there means "scored" (only
  `pipeline/retrieval_eval.py` passes a gold answer). The code is kept as is
  to reproduce the released graph, and `tests/test_capture.py` marks the test
  `xfail`. `--full-narrowing` sizes the space and `kgp:outputCardinality` by
  the scored pool and lists the members below the top-k as excluded.
