"""End-to-end pipeline runner.

Wires loaders → chunking → four phases → provenance serialization →
CQ report. The phases call Bedrock; with --replay every Bedrock call is
served from the response log and none is issued.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml

# Minimal .env loader (for AWS credentials) so python-dotenv is not a
# dependency.
def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


_load_dotenv()

from rdflib import Graph, Literal, RDF

from kgprov import Recorder
from kgprov.capture import KGP, PROV, BASE
from kgprov.llm import BedrockClient
from .chunking import chunk_document
from .loaders import Document, load_bc5cdr, parse_mesh_xml
from .phases import (
    Mention, Relationship, KGEdge, Phase2ArmDecision, RejectedSpan,
    phase1_extract, phase2_ground, phase3_extract_relations,
    phase4_normalize, register_agents,
)
from .registry import EscalationCache, RegistryStats
from .retrieval import MeshIndex, RetrievalParams
from . import sampling


# Per-token list prices for the cost estimate (USD per 1k tokens). Display
# only; billing is via AWS, and the invoice is the authority.
_PRICE = {
    "haiku": (0.001, 0.005),      # Claude Haiku 4.5 list
    "sonnet": (0.003, 0.015),     # Claude Sonnet 4.5 list
    "unknown": (0.0, 0.0),
}


def _price_for(model_id: str) -> tuple[float, float]:
    m = model_id.lower()
    if "haiku" in m:
        return _PRICE["haiku"]
    if "sonnet" in m:
        return _PRICE["sonnet"]
    return _PRICE["unknown"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply --override key=value from CLI (simple dot-path)."""
    for kv in overrides:
        if "=" not in kv:
            continue
        path, value = kv.split("=", 1)
        parts = path.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        # coerce
        try:
            v: object = int(value)
        except ValueError:
            try:
                v = float(value)
            except ValueError:
                v = value if not isinstance(value, str) else \
                    value.lower() in {"true", "yes", "1"} \
                    if value.lower() in {"true", "false", "yes", "no", "1", "0"} \
                    else value
        node[parts[-1]] = v
    return cfg


# ---------------------------------------------------------------------
#  CSV writers
# ---------------------------------------------------------------------

_MENTION_COLS = [
    "entity_id", "doc_id", "chunk_id", "start_char", "end_char",
    "doc_start_char", "raw_span", "resolved_string", "resolution_type",
    "semantic_category", "cui", "mapping_tier", "mapping_confidence",
    "candidate_score",
]

_RELATION_COLS = [
    "relation_id", "doc_id", "chunk_id", "subject_entity_id",
    "object_entity_id", "subject_cui", "object_cui",
    "raw_action_phrase", "action_start_char", "action_end_char",
    "polarity", "modality", "conditions", "is_provisional",
]

_KG_COLS = [
    "subject_cui", "normalized_predicate", "object_cui", "polarity",
    "modality", "conditions", "negated", "evidence_count",
    "supporting_relation_ids", "is_provisional",
]

_ARM_COLS = [
    "dedup_norm", "dedup_category", "arm",
    # outcome is authoritative; mapped_id alone cannot distinguish "the
    # model declined" from "no model was called".
    "outcome", "mapped_id", "confidence",
    # escalated = the unbounded cell ran; escalation_triggered = Tier 1 met
    # the trigger. They differ for an arm running bounded-only.
    "candidate_ids", "escalated", "escalation_triggered",
    # Makes the mention->decision join exact instead of reconstructed by
    # string-matching Phase 1's rewrites against gold surface forms.
    "covered_entity_ids", "n_covered",
    "tier2_tool_calls", "tier2_termination",
    "tier2_claim", "tier2_verified", "tier2_checks",
    # The escalation cache bypasses the LLM client entirely, so a hit is
    # absent from the response log and from cost_usd_notional. A
    # response-log replay is not: notional prices those.
    "tier2_from_cache",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------
#  Fragmented provenance serialization
# ---------------------------------------------------------------------

def _flush_fragment(recorder: Recorder, fragment_path: Path,
                    ontology_versions: list[str]) -> int:
    """Serialize the current recorder graph and reset it.

    Content-addressed IDs mean cross-fragment references resolve to the
    same IRIs at merge time. Run-level metadata + agents + ontology
    snapshots are re-registered into the new empty graph so each
    fragment is self-contained.
    """
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    n_before = len(recorder)
    recorder.serialize(str(fragment_path))
    print(f"[flush] wrote fragment {fragment_path.name} "
          f"({n_before} triples)", flush=True)

    # Fresh graph, re-bind namespaces
    g = Graph()
    g.bind("kgp", KGP)
    g.bind("prov", PROV)
    g.bind("run", BASE)
    # Run-level metadata (subset — omit generatedAtTime to avoid drift)
    g.add((recorder.run_uri, RDF.type, KGP.PipelineRun))
    g.add((recorder.run_uri, RDF.type, PROV.Bundle))
    g.add((recorder.run_uri, KGP.configurationVariantLabel,
           Literal(recorder.config_variant)))
    g.add((recorder.run_uri, KGP.modelId, Literal(recorder.model_id)))
    g.add((recorder.run_uri, KGP.ontologyVersion,
           Literal(recorder.ontology_version)))
    recorder.g = g
    recorder._spaces = {}

    # Re-register agents + ontology snapshots so each fragment stands alone.
    from .phases import register_agents
    register_agents(recorder)
    for v in ontology_versions:
        recorder.ontology_snapshot(f"mesh_{v}", v)
    return n_before


def _merge_fragments(fragment_paths: list[Path], out_path: Path) -> int:
    """Load and re-serialize the union of fragment TTLs (rdflib dedupes)."""
    merged = Graph()
    merged.bind("kgp", KGP)
    merged.bind("prov", PROV)
    merged.bind("run", BASE)
    for p in fragment_paths:
        print(f"[merge] parsing {p.name} ({p.stat().st_size} B)", flush=True)
        merged.parse(str(p), format="turtle")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[merge] serializing {len(merged)} triples -> {out_path}",
          flush=True)
    merged.serialize(str(out_path), format="turtle")
    return len(merged)


# ---------------------------------------------------------------------
#  Pipeline
# ---------------------------------------------------------------------

def run_pipeline(cfg: dict, args) -> dict:
    run_id = args.run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    variant = args.variant or "default"

    out_root = Path("out")
    results_dir = out_root / "results" / run_id
    prov_path = out_root / "provenance" / f"{run_id}.ttl"
    response_log = Path(cfg["paths"]["response_log"]) / run_id
    cq_report_dir = out_root / "cq_report"
    for d in [results_dir, prov_path.parent, response_log, cq_report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Data
    print(f"[run] loading BC5CDR split={args.split}")
    docs = load_bc5cdr(cfg["corpus"]["bc5cdr_dir"], args.split)
    if args.n_docs:
        docs = docs[:args.n_docs]
    print(f"[run] {len(docs)} documents")

    # ---- Ontology + index
    print(f"[run] loading MeSH primary {cfg['ontology']['primary']}")
    ontology = parse_mesh_xml(cfg["ontology"]["primary"],
                              cfg["ontology"]["version_primary"])
    prior = None
    if cfg["ontology"].get("prior") and not args.no_prior:
        print(f"[run] loading MeSH prior {cfg['ontology']['prior']}")
        prior = parse_mesh_xml(cfg["ontology"]["prior"],
                               cfg["ontology"]["version_prior"])

    index_cache = Path(f"data/mesh/index_{cfg['ontology']['version_primary']}")
    index = MeshIndex.build_or_load(ontology, index_cache)

    retr_params = RetrievalParams(
        dense_top_k=cfg["retrieval"]["dense_top_k"],
        sparse_top_k=cfg["retrieval"]["sparse_top_k"],
        rrf_k=cfg["retrieval"]["rrf_k"],
        fused_top_k=cfg["retrieval"]["fused_top_k"],
        rerank_top_k=cfg["retrieval"]["rerank_top_k"],
        score_floor=cfg["retrieval"]["score_floor"],
    )

    # ---- LLM
    weak_model = cfg["models"]["weak_arm"]["model_id"]
    strong_model = cfg["models"]["strong_arm"]["model_id"]
    llm = BedrockClient(
        region=cfg["aws"]["region"],
        response_log_dir=str(response_log),
        replay_mode=args.replay,
        max_retries=cfg["run"]["max_retries"])

    # ---- Recorder
    recorder = Recorder(
        run_id=run_id,
        config_variant=variant,
        model_id=f"weak={weak_model};strong={strong_model}",
        ontology_version=cfg["ontology"]["version_primary"],
    )
    register_agents(recorder)
    recorder.ontology_snapshot(
        f"mesh_{cfg['ontology']['version_primary']}",
        cfg["ontology"]["version_primary"])
    if prior is not None:
        recorder.ontology_snapshot(
            f"mesh_{cfg['ontology']['version_prior']}",
            cfg["ontology"]["version_prior"])

    # ---- Ablation-only tweaks
    tier1_arms = []
    for arm_name in args.tier1_arms:
        if arm_name == "weak_arm":
            tier1_arms.append(("weak_arm", weak_model))
        elif arm_name == "strong_arm":
            tier1_arms.append(("strong_arm", strong_model))
        else:
            raise ValueError(f"unknown arm: {arm_name}")
    primary_arm = args.primary_arm
    if primary_arm not in [a for a, _ in tier1_arms]:
        # mention.cui comes from this arm's decision; without it every
        # mention falls through to a provisional identifier and the run
        # measures nothing.
        raise ValueError(
            f"--primary-arm {primary_arm} is not in --tier1-arms "
            f"{args.tier1_arms}")

    # Which arms run the unbounded cell. Decoupled from --tier1-arms
    # because the strong arm's Tier 2 is the most expensive step in the
    # pipeline, so its bounded cell can be run on its own first.
    if args.tier2_arms is None:
        tier2_arms = [a for a, _ in tier1_arms]
    else:
        tier2_arms = [a for a in args.tier2_arms if a != "none"]
        if "none" in args.tier2_arms and tier2_arms:
            raise ValueError("--tier2-arms none excludes naming arms")
    unknown_t2 = set(tier2_arms) - {a for a, _ in tier1_arms}
    if unknown_t2:
        raise ValueError(
            f"--tier2-arms {sorted(unknown_t2)} not in --tier1-arms "
            f"{args.tier1_arms}: escalation is triggered by a Tier 1 "
            f"failure, so an arm needs a Tier 1 decision to escalate from")

    # ---- Strong-arm sampling
    # The strong cells cost 3x the weak ones per token (_PRICE), so only
    # they are sampled. Sampling is per dedup key, seeded, and
    # order-independent, so a resumed run samples the identical subset and
    # replays it from the response log.
    sampler = None
    sample_summary = None
    if args.sample_strong:
        if "strong_arm" not in {a for a, _ in tier1_arms}:
            raise ValueError(
                "--sample-strong has nothing to sample: strong_arm is not "
                "in --tier1-arms")
        keys_per_doc = float(cfg.get("run", {}).get(
            "keys_per_doc_estimate", sampling.DEFAULT_KEYS_PER_DOC))
        rate = (args.sample_rate if args.sample_rate is not None
                else sampling.rate_for_target(
                    args.sample_strong, len(docs), keys_per_doc))
        sampler = sampling.ArmSampler(
            arms={"strong_arm"}, rate=rate,
            seed=args.sample_seed, cap=args.sample_strong)
        print(f"[run] sampling strong_arm: target {args.sample_strong} keys, "
              f"rate {rate:.4f}, seed {args.sample_seed} "
              f"(projected {len(docs) * keys_per_doc:.0f} keys)")

    all_mentions: list[Mention] = []
    all_rejected: list[RejectedSpan] = []
    # Identifiers the model emitted more than once. Not rejections -- the
    # span was valid and was produced twice -- but every count taken off the
    # CSVs is wrong by this much if the collapse is silent.
    all_duplicates: list[str] = []
    all_relations: list[Relationship] = []
    all_arm_decisions: list[Phase2ArmDecision] = []
    retrieval_cache: dict = {}

    # Tier 2 escalation state. The cache is persistent and shared across
    # runs: web escalation is the most expensive operation here and its
    # results are stable, so a hit short-circuits. A miss always runs the
    # full agent. The path is settable so a cost-measuring run can start
    # from a cold cache without discarding the warm one: a hit costs
    # nothing and is not counted in the cost figures.
    escalation_cache = EscalationCache(
        Path(args.escalation_cache) if args.escalation_cache
        else out_root / "escalation_cache.json")
    registry_stats = RegistryStats()
    tier2_budget = int(cfg.get("run", {}).get("tier2_max_tool_calls", 6))

    # Fragmented provenance: flush every N docs to cap memory.
    frag_dir = out_root / "provenance" / f"{run_id}_fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    frag_flush_every = args.frag_flush_every
    fragment_paths: list[Path] = []
    ontology_versions = [cfg["ontology"]["version_primary"]]
    if prior is not None:
        ontology_versions.append(cfg["ontology"]["version_prior"])

    t0 = time.time()
    for di, doc in enumerate(docs, start=1):
        chunks = chunk_document(
            doc.full_text,
            max_chunk_tokens=cfg["chunking"]["max_chunk_tokens"],
            sentence_overlap=cfg["chunking"]["sentence_overlap"])
        if not chunks:
            continue

        # ---- Phase 1
        mentions, rejected = phase1_extract(
            doc, chunks, llm=llm, model_id=weak_model, recorder=recorder,
            temperature=cfg["decoding"]["temperature"],
            max_tokens=cfg["models"]["weak_arm"]["max_tokens"],
            duplicates_out=all_duplicates)
        all_mentions.extend(mentions)
        all_rejected.extend(rejected)

        # ---- Gold annotations: attach BC5CDR gold to any pipeline mention
        # whose doc-relative span overlaps a gold mention with a MeSH ID.
        # CQ2 joins the gold value to constraint-space members.
        for m in mentions:
            m_start = m.doc_start_char
            m_end = m.doc_start_char + (m.end_char - m.start_char)
            for gold in doc.mentions:
                if gold.mesh_id is None:
                    continue
                if max(gold.start, m_start) < min(gold.end, m_end):
                    recorder.gold_annotation(
                        f"gold_{m.entity_id}",
                        assesses=m.entity_id,
                        value=gold.mesh_id)
                    break  # first overlap wins

        # ---- Phase 2
        mentions, arm_decs = phase2_ground(
            mentions, index=index, params=retr_params,
            ontology=ontology, ontology_prior=prior,
            llm=llm, tier1_arms=tier1_arms, primary_arm=primary_arm,
            recorder=recorder,
            temperature=cfg["decoding"]["temperature"],
            max_tokens=cfg["models"]["weak_arm"]["max_tokens"],
            retrieval_cache=retrieval_cache,
            escalation_cache=escalation_cache,
            registry_stats=registry_stats,
            tier2_max_tool_calls=tier2_budget,
            tier2_arms=tier2_arms,
            sampler=sampler,
            compact_narrowing=not args.full_narrowing)
        all_arm_decisions.extend(arm_decs)

        # ---- Phase 3
        rels = phase3_extract_relations(
            doc, chunks, mentions,
            llm=llm, model_id=weak_model, recorder=recorder,
            temperature=cfg["decoding"]["temperature"],
            max_tokens=cfg["models"]["weak_arm"]["max_tokens"])
        all_relations.extend(rels)

        if di % 5 == 0 or di == len(docs):
            elapsed = time.time() - t0
            print(f"[run] doc {di}/{len(docs)} ({elapsed:.1f}s) "
                  f"mentions={len(all_mentions)} relations={len(all_relations)} "
                  f"triples_live={len(recorder)}", flush=True)

        # Fragment flush + memory reset every N docs.
        if frag_flush_every > 0 and di % frag_flush_every == 0 and di != len(docs):
            frag_path = frag_dir / f"frag_{len(fragment_paths):03d}.ttl"
            _flush_fragment(recorder, frag_path, ontology_versions)
            fragment_paths.append(frag_path)

    # ---- Phase 4 (batch over all relations)
    print(f"[run] Phase 4 over {len(all_relations)} relations")
    edges = phase4_normalize(
        all_relations, all_mentions, llm=llm, model_id=weak_model,
        recorder=recorder,
        batch_size=50,
        temperature=cfg["decoding"]["temperature"],
        max_tokens=cfg["models"]["weak_arm"]["max_tokens"])

    # ---- Persist tables
    print(f"[run] writing tables to {results_dir}")
    _write_csv(results_dir / "mention_table.csv", _MENTION_COLS,
               [{k: getattr(m, k) for k in _MENTION_COLS} for m in all_mentions])
    _write_csv(results_dir / "relationship_table.csv", _RELATION_COLS,
               [{**{k: getattr(r, k) for k in _RELATION_COLS if k != "conditions"},
                 "conditions": json.dumps(r.conditions)} for r in all_relations])
    _write_csv(results_dir / "kg_table.csv", _KG_COLS,
               [{**{k: getattr(e, k) for k in _KG_COLS
                    if k not in ("conditions", "supporting_relation_ids")},
                 "conditions": json.dumps(e.conditions),
                 "supporting_relation_ids": "|".join(e.supporting_relation_ids)}
                for e in edges])
    _write_csv(results_dir / "rejected_spans.csv",
               ["doc_id", "chunk_id", "raw_span", "reason"],
               [asdict(r) for r in all_rejected])
    _write_csv(results_dir / "tier1_arm_decisions.csv", _ARM_COLS,
               [{
                   "dedup_norm": d.dedup_key[0],
                   "dedup_category": d.dedup_key[1],
                   "arm": d.arm,
                   "outcome": d.outcome,
                   "mapped_id": d.mapped_id or "",
                   "confidence": d.confidence,
                   "candidate_ids": "|".join(d.candidate_ids),
                   "escalated": d.escalated,
                   "escalation_triggered": d.escalation_triggered,
                   "covered_entity_ids": "|".join(d.covered_entity_ids),
                   "n_covered": len(d.covered_entity_ids),
                   "tier2_tool_calls": d.tier2_tool_calls,
                   "tier2_termination": d.tier2_termination,
                   "tier2_claim": d.tier2_claim,
                   "tier2_verified": d.tier2_verified,
                   "tier2_checks": d.tier2_checks,
                   "tier2_from_cache": d.tier2_from_cache,
               } for d in all_arm_decisions])

    # The sample as a positive record. Without it, "this arm has no row for
    # this key" is indistinguishable from a dropped row.
    if sampler is not None:
        sample_summary = sampler.summary()
        _write_csv(
            results_dir / "arm_sample.csv",
            ["arm", "dedup_norm", "dedup_category"],
            [{"arm": arm, "dedup_norm": k[0], "dedup_category": k[1]}
             for arm in sample_summary["sampled_arms"]
             for k in sorted(sampler.admitted(arm))])

    # ---- Provenance: flush the final fragment and merge everything.
    frag_path = frag_dir / f"frag_{len(fragment_paths):03d}.ttl"
    _flush_fragment(recorder, frag_path, ontology_versions)
    fragment_paths.append(frag_path)

    if len(fragment_paths) == 1:
        # Single fragment: no merge needed, just copy.
        print(f"[run] one fragment — copying to {prov_path}", flush=True)
        import shutil
        shutil.copyfile(fragment_paths[0], prov_path)
        triples = 0
        try:
            from rdflib import Graph as _G
            g = _G(); g.parse(str(prov_path), format="turtle")
            triples = len(g)
        except Exception:
            pass
    else:
        print(f"[run] merging {len(fragment_paths)} fragments -> "
              f"{prov_path}", flush=True)
        triples = _merge_fragments(fragment_paths, prov_path)

    # ---- Cost
    # Price each model at its own rate. A single blended rate is wrong once
    # two arms are used.
    per_model_cost: dict[str, dict] = {}
    notional_usd = billed_usd = 0.0
    for mid, t in llm.by_model.items():
        pi, po = _price_for(mid)
        notional = round(t["input"] / 1000 * pi + t["output"] / 1000 * po, 4)
        billed = round(t["billed_input"] / 1000 * pi +
                       t["billed_output"] / 1000 * po, 4)
        per_model_cost[mid] = {
            **t, "notional_usd": notional, "billed_usd": billed}
        notional_usd += notional
        billed_usd += billed
    notional_usd = round(notional_usd, 4)
    billed_usd = round(billed_usd, 4)
    p_in, p_out = _price_for(weak_model)
    weak_cost = llm.cost_summary(p_in, p_out)

    # Split the spend by how it scales, because the two halves extrapolate
    # differently and a single per-document rate is wrong the moment one
    # cell is a fixed-size sample. Sampled arms are identified by model id;
    # that holds while each sampled arm has its own model, which is the
    # only configuration --sample-strong permits.
    sampled_models = {"strong_arm": strong_model}
    sampled_ids = {sampled_models[a]
                   for a in (sample_summary or {}).get("sampled_arms", [])
                   if a in sampled_models}
    sampled_usd = round(sum(v["notional_usd"] for k, v in
                            per_model_cost.items() if k in sampled_ids), 4)
    census_usd = round(notional_usd - sampled_usd, 4)

    # ---- Manifest
    wall = round(time.time() - t0, 1)
    manifest = {
        "run_id": run_id,
        "config_variant": variant,
        "split": args.split,
        "n_docs": len(docs),
        "weak_arm": weak_model,
        "strong_arm": strong_model,
        "tier1_arms": [a[0] for a in tier1_arms],
        "tier2_arms": tier2_arms,
        "primary_arm": primary_arm,
        # None = census. Carries the seed, the rate, the cap and the
        # realised key counts, so the sample is reproducible from the
        # manifest alone and its size is reported rather than assumed.
        "sampling": sample_summary,
        "ontology_primary": cfg["ontology"]["version_primary"],
        "ontology_prior": cfg["ontology"]["version_prior"] if prior else None,
        "retrieval_params": {
            "dense_top_k": retr_params.dense_top_k,
            "sparse_top_k": retr_params.sparse_top_k,
            "rrf_k": retr_params.rrf_k,
            "fused_top_k": retr_params.fused_top_k,
            "rerank_top_k": retr_params.rerank_top_k,
            "score_floor": retr_params.score_floor,
        },
        "wall_clock_sec": wall,
        # What the experiment costs from a cold cache. This is the figure
        # to report as the cost of the run.
        "cost_usd_notional": notional_usd,
        # What this invocation actually spent. Differs from notional by
        # exactly the replayed calls; a resumed run is mostly cache.
        "cost_usd_billed": billed_usd,
        # The census half scales with documents; the sampled half is capped
        # by the sample size and does not. Extrapolating the sum at one
        # per-document rate over-counts the sampled half.
        "cost_usd_census": census_usd,
        "cost_usd_sampled": sampled_usd,
        "n_calls_live": llm.n_calls_live,
        "n_calls_cached": llm.n_calls_cached,
        # Escalations served from the persistent escalation cache. These
        # are not in either cost figure: the cache short-circuits ahead of
        # the LLM client, so no request is logged and no tokens are counted.
        # A non-zero count means notional is a lower bound.
        "n_escalations": sum(1 for d in all_arm_decisions if d.escalated),
        "n_escalations_from_cache": sum(
            1 for d in all_arm_decisions if d.tier2_from_cache),
        "cost_by_model": per_model_cost,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "triples": triples,
        # Distinct mentions, after collapsing repeated identifiers.
        "mentions": len(all_mentions),
        "duplicate_mentions_collapsed": len(all_duplicates),
        "relations": len(all_relations),
        "kg_edges": len(edges),
        "rejected_spans": len(all_rejected),
        "replay_mode": args.replay,
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[run] manifest: {json.dumps(manifest, indent=2)}")

    # ---- CQ report
    try:
        print(f"[run] running CQ report against {prov_path}")
        cq_out = subprocess.run(
            [sys.executable, "tools/cq_queries.py", str(prov_path)],
            capture_output=True, text=True, check=False)
        report = cq_report_dir / f"{run_id}.txt"
        report.write_text((cq_out.stdout or "") + "\n---STDERR---\n" +
                          (cq_out.stderr or ""))
        print(f"[run] cq report: {report}")
    except Exception as e:
        print(f"[run] CQ report failed: {e}")

    return manifest


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="key.path=value")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--no-prior", action="store_true",
                    help="Skip loading prior MeSH (faster smoke)")
    ap.add_argument("--frag-flush-every", type=int, default=50,
                    help="Flush provenance to disk and reset in-memory "
                    "graph every N docs (0 disables).")
    ap.add_argument("--tier1-arms", nargs="+",
                    default=["weak_arm", "strong_arm"],
                    choices=["weak_arm", "strong_arm"])
    ap.add_argument("--tier2-arms", nargs="+", default=None,
                    choices=["weak_arm", "strong_arm", "none"],
                    help="Arms that run the unbounded cell. Must be a "
                    "subset of --tier1-arms; defaults to all of them. "
                    "'none' runs the bounded cells only.")
    ap.add_argument("--primary-arm", default="weak_arm",
                    choices=["weak_arm", "strong_arm"])
    ap.add_argument("--sample-strong", type=int, default=None, metavar="N",
                    help="Run the strong arm on at most N dedup keys, "
                    "sampled by seeded hash, instead of a census. Applies "
                    "to both of its cells: skipping an arm's Tier 1 for a "
                    "key skips its Tier 2, so the strong row of the 2x2 "
                    "sits on one population.")
    ap.add_argument("--sample-rate", type=float, default=None,
                    help="Per-key admission rate. Default: N divided by "
                    "the projected key count (n_docs x "
                    "run.keys_per_doc_estimate).")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="Logged in the manifest; the sample is "
                    "reproducible from it.")
    ap.add_argument("--full-narrowing", action="store_true",
                    help="Record the full member list of every intermediate "
                    "constraint space instead of an empty stand-in. The "
                    "compact default keeps cardinalities and exclusion "
                    "reasons -- enough for the aggregate CQ3 -- at a small "
                    "fraction of the triple cost, but leaves kgp:excluded "
                    "pointing at placeholders, so the emitted graph cannot "
                    "say which stage removed a *particular* answer. This "
                    "flag makes that queryable, at roughly an order of "
                    "magnitude more triples. Changes capture only: prompts "
                    "and responses are identical, so with a warm "
                    "response log no Bedrock calls are made.")
    ap.add_argument("--escalation-cache", default=None, metavar="PATH",
                    help="Escalation cache file (default "
                    "out/escalation_cache.json). Point at an unused path "
                    "to measure Tier 2 cold: a cache hit bypasses the LLM "
                    "client, so it never enters cost_usd_notional.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    run_pipeline(cfg, args)


if __name__ == "__main__":
    _main()
