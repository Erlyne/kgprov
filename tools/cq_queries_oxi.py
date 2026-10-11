#!/usr/bin/env python3
"""
Pyoxigraph runner for the competency questions.

rdflib's Python-based SPARQL engine goes quadratic on large graphs and
becomes unusable at ~700k triples. Pyoxigraph is a native (Rust) SPARQL
engine that runs the same queries at that size.

Query text is imported unchanged from cq_queries.py. This module changes
only how the graph is loaded and queried; it also loads kgprov.ttl
alongside the graph (see load_graph).

Run:
    python tools/cq_queries_oxi.py out/provenance/stage1_weak_clean.ttl
    python tools/cq_queries_oxi.py out/provenance/stage1_weak_clean.ttl --cq CQ3
    python tools/cq_queries_oxi.py out/provenance/stage1_weak_clean.ttl \\
        --out cq_report.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the query text unchanged.
from tools.cq_queries import CQS, PREFIXES  # noqa: E402

from pyoxigraph import Store, RdfFormat  # noqa: E402


# ---------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------

ONTOLOGY = Path(__file__).resolve().parents[1] / "kgprov" / "ontology" / "kgprov.ttl"


def load_graph(ttl_path: str) -> Store:
    """Bulk-load a Turtle file, plus the vocabulary, into a Store.

    The vocabulary rides along for the same reason validate_shapes.py
    loads it: a competency question is asked of a provenance graph read
    under its profile. The kgp:RejectionReason and kgp:AttestationOutcome
    individuals are typed in kgprov.ttl and nowhere else, so without it a
    query can name the individual but not the class it belongs to.
    """
    store = Store()
    t0 = time.time()
    # bulk_load is the fast path for large single-file loads.
    with open(ttl_path, "rb") as f:
        store.bulk_load(f, RdfFormat.TURTLE)
    n_data = len(store)
    with open(ONTOLOGY, "rb") as f:
        store.bulk_load(f, RdfFormat.TURTLE)
    elapsed = time.time() - t0
    print(f"loaded {n_data:,} triples from {ttl_path} "
          f"+ {len(store) - n_data:,} from the vocabulary "
          f"in {elapsed:.1f}s", flush=True)
    return store


# ---------------------------------------------------------------------
#  Formatting solutions
# ---------------------------------------------------------------------

def _term_repr(term) -> str:
    """Compact human-readable form of an oxigraph term.

    Uses .value where available (Literal / NamedNode / BlankNode), which
    strips the surrounding quotes / angle brackets / datatype suffix.
    """
    if term is None:
        return ""
    if hasattr(term, "value"):
        return term.value
    return str(term)


def run_one(store: Store, cid: str, verbose: bool = True) -> list[dict]:
    """Run one CQ and return its solutions as a list of {var: term_string}."""
    spec = CQS[cid]
    sparql = PREFIXES + spec["sparql"]
    t0 = time.time()
    solutions_iter = store.query(sparql)
    # `.variables` is on QuerySolutions (the iterator), listing projected
    # variables in select order.
    var_names = [str(v).lstrip("?") for v in solutions_iter.variables]
    n_vars = len(var_names)
    rows: list[dict] = []
    for sol in solutions_iter:
        row = {}
        for i, name in enumerate(var_names):
            row[name] = _term_repr(sol[i]) if i < n_vars else ""
        rows.append(row)
    elapsed = time.time() - t0

    if verbose:
        bucket = spec["bucket"]
        tag = {"A": "core PROV", "B": "extension",
               "C": "out of scope"}[bucket]
        print(f"\n{'='*70}")
        print(f"{cid}  [{bucket} - {tag}]  ({elapsed:.2f}s)")
        print(f"  Q: {spec['question']}")
        print(f"{'-'*70}")
        if not rows:
            print("  (no rows - unpopulated or inexpressible)")
        else:
            cols = list(rows[0].keys())
            header = "  " + "  ".join(f"{c:>24}" for c in cols)
            print(header)
            for r in rows:
                print("  " + "  ".join(f"{r[c]:>24}" for c in cols))
        print(f"  -> {spec['demonstrates']}")
    return rows


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--cq", default=None, help="run a single CQ, e.g. CQ3")
    ap.add_argument("--out", default=None,
                    help="Write full report to this file instead of stdout")
    args = ap.parse_args()

    # Tee output to file if requested, but also to console for progress.
    buf: io.StringIO | None = None
    if args.out:
        buf = io.StringIO()

    def _do_run() -> tuple[int, int]:
        store = load_graph(args.graph)
        targets = [args.cq] if args.cq else list(CQS.keys())
        empty: list[str] = []
        for cid in targets:
            rows = run_one(store, cid)
            if not rows:
                empty.append(cid)
        print(f"\n{'='*70}")
        print(f"ran {len(targets)} queries; "
              f"{len(targets) - len(empty)} returned rows, "
              f"{len(empty)} empty")
        if empty:
            print(f"EMPTY (investigate): {', '.join(empty)}")
        else:
            print("no query returned zero rows. Ten of the thirteen "
                  "queries carry a stated limitation in "
                  "tools/cq_queries.py, which this runner does not print.")
        return len(targets), len(empty)

    if buf is not None:
        # Capture into buf AND echo to real stdout live.
        class _Tee(io.TextIOBase):
            def __init__(self, *streams):
                self._streams = streams
            def write(self, s):
                for s2 in self._streams:
                    s2.write(s); s2.flush()
                return len(s)
            def flush(self):
                for s2 in self._streams:
                    s2.flush()
        with redirect_stdout(_Tee(sys.__stdout__, buf)):
            _do_run()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.__stderr__)
    else:
        _do_run()


if __name__ == "__main__":
    main()
