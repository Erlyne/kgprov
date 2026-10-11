"""Live MeSH registry access for the Tier 2 escalation agent.

Two separate responsibilities:

  1. ``search_biomedical_web`` -- the tool the agent may call. An ordinary
     deterministic HTTP request against NCBI E-utilities. The agency is
     not here; it is in the model deciding whether to call this, what
     query string to compose, how to read the result, and whether to call
     again.

  2. ``resolve_live`` -- existence verification against the registry, for
     identifiers absent from the local ontology.

Why (2) exists at all: an identifier returned by a web agent is a *claim*
about an external registry, and MeSH identifiers are format-regular
enough that a well-formed string is no evidence of existence. ``D999999``
is as fluent as ``D008687``. Identifiers absent from the local table are
therefore checked against the live registry.

E-utilities needs no API key and permits 3 requests/second. A key raises
that to 10; set NCBI_API_KEY if you have one.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TOOL = "kgprov-thesis"
_UA = "KGPROV-thesis/1.0 (academic research; provenance study)"
#: 3 req/s is the anonymous ceiling. With NCBI_API_KEY set it is 10, but
#: the interval below stays conservative: being throttled mid-run costs
#: more than the time saved.
_MIN_INTERVAL = 0.34
_MESH_UI = re.compile(r"^[CD]\d{6,9}$")

_last_call = [0.0]


def _throttle() -> None:
    """NCBI blocks abusers. Serialise and space the calls."""
    gap = time.time() - _last_call[0]
    if gap < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - gap)
    _last_call[0] = time.time()


def _get(path: str, params: dict, timeout: float = 20.0) -> str:
    # NCBI's usage policy asks programmatic clients to identify themselves
    # with `tool` and `email`. Sustained anonymous volume can be throttled
    # or blocked.
    params = {**params, "tool": _TOOL}
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    key = os.environ.get("NCBI_API_KEY")
    if key:
        # A key raises the ceiling from 3 to 10 requests/second.
        params["api_key"] = key
    url = f"{_BASE}/{path}?" + urllib.parse.urlencode(params)
    _throttle()
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def is_wellformed(ui: str) -> bool:
    """Shape check only. Says nothing about existence."""
    return bool(_MESH_UI.match((ui or "").strip()))


# ---------------------------------------------------------------------
#  The tool the agent may call
# ---------------------------------------------------------------------

@dataclass
class RegistryStats:
    """Counters for the provenance record and the cost report."""
    searches: int = 0
    resolves: int = 0
    errors: int = 0
    urls: list[str] = field(default_factory=list)


def search_biomedical_web(query: str, *, retmax: int = 5,
                          stats: RegistryStats | None = None) -> str:
    """Query the live MeSH database and return a readable digest.

    Deterministic given the query; the agent's contribution is the query.
    Returns text rather than JSON because the payload is going back into a
    model context, and a failure is returned as text rather than raised so
    the agent can react to it instead of the run dying.
    """
    if stats:
        stats.searches += 1
    try:
        raw = _get("esearch.fcgi", {"db": "mesh", "term": query,
                                    "retmode": "json", "retmax": retmax})
        ids_ = json.loads(raw).get("esearchresult", {}).get("idlist", [])
        if stats:
            stats.urls.append(f"{_BASE}/esearch.fcgi?db=mesh&term="
                              + urllib.parse.quote(query))
        if not ids_:
            return "No registry ID found."

        raw = _get("esummary.fcgi", {"db": "mesh", "id": ",".join(ids_),
                                     "retmode": "json"})
        res = json.loads(raw).get("result", {})
        lines = []
        for uid in res.get("uids", []):
            rec = res.get(uid, {})
            name = rec.get("ds_meshterms", [None])[0] or rec.get("title", "?")
            ui = rec.get("ds_meshui") or ""
            scope = (rec.get("ds_scopenote") or "")[:200]
            lines.append(f"- {ui or '(no UI)'}  {name}"
                         + (f"\n    {scope}" if scope else ""))
        return ("Registry results:\n" + "\n".join(lines)) if lines \
            else "No registry ID found."
    except Exception as e:                              # noqa: BLE001
        if stats:
            stats.errors += 1
        return f"Search failed ({type(e).__name__}). No registry ID found."


# ---------------------------------------------------------------------
#  Existence verification
# ---------------------------------------------------------------------

@dataclass
class LiveRecord:
    ui: str
    preferred_name: str
    synonyms: list[str]
    exists: bool
    source_url: str


_resolve_cache: dict[str, LiveRecord | None] = {}


def resolve_live(ui: str, *, stats: RegistryStats | None = None
                 ) -> LiveRecord | None:
    """Look an identifier up in the live registry. None means absent.

    Cached in-process because the same fabricated identifier tends to
    recur.
    """
    ui = (ui or "").strip()
    if not is_wellformed(ui):
        return None
    if ui in _resolve_cache:
        return _resolve_cache[ui]
    if stats:
        stats.resolves += 1

    url = f"https://id.nlm.nih.gov/mesh/{ui}"
    try:
        raw = _get("esearch.fcgi", {"db": "mesh", "term": f"{ui}[MeSH Unique ID]",
                                    "retmode": "json"})
        ids_ = json.loads(raw).get("esearchresult", {}).get("idlist", [])
        if not ids_:
            _resolve_cache[ui] = None
            return None
        raw = _get("esummary.fcgi", {"db": "mesh", "id": ids_[0],
                                     "retmode": "json"})
        rec = json.loads(raw).get("result", {}).get(ids_[0], {})
        terms = rec.get("ds_meshterms") or []
        out = LiveRecord(
            ui=rec.get("ds_meshui") or ui,
            preferred_name=terms[0] if terms else (rec.get("title") or ""),
            synonyms=list(terms[1:]),
            exists=True, source_url=url)
    except Exception:                                   # noqa: BLE001
        if stats:
            stats.errors += 1
        # An unreachable registry is not evidence of absence, so the result
        # is not cached. The caller cannot tell this None from absence: the
        # EXISTENCE check fails and the outcome is recorded as Refuted.
        return None
    _resolve_cache[ui] = out
    return out


# ---------------------------------------------------------------------
#  Persistent escalation cache
# ---------------------------------------------------------------------

class EscalationCache:
    """Disk cache keyed on the deduplication tuple.

    Escalation is the most expensive operation in the pipeline and its
    results are stable across runs, so a hit short-circuits. This is a
    cost mechanism only: a miss always runs the full agent. It never
    substitutes a lookup for inference.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._d: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._d = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._d = {}

    @staticmethod
    def _key(norm_text: str, category: str, arm: str) -> str:
        return f"{arm}|{category}|{norm_text}"

    def get(self, norm_text: str, category: str, arm: str) -> dict | None:
        return self._d.get(self._key(norm_text, category, arm))

    def put(self, norm_text: str, category: str, arm: str,
            payload: dict) -> None:
        self._d[self._key(norm_text, category, arm)] = payload
        self.path.write_text(json.dumps(self._d, indent=1), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._d)
