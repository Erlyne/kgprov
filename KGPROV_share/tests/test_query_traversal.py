"""The annotations must be backed by the queries they name.

The pruning criterion acts on the SPARQL, not on the prose statement of a
question: a term is kept for a question only if that question's SPARQL
names it. ``prune_profile`` reads the hand-written ``kgp:motivatedByCQ``
annotations, so these tests compare them with the query text.

Three directions are checked:

* every active annotation is backed by a mention in that query;
* every retained term is reached by at least one active query;
* no query names a ``kgp:`` term the ontology does not declare.
"""

from __future__ import annotations

import re

import pytest

from tools.cq_queries import CQS, terms_named, terms_named_optionally
from tools.prune_profile import DEFAULT_ACTIVE, classify

#: Terms exempted from the checks below, mapped to the reason. An entry
#: should name what must change in the graph to remove it. Empty.
KNOWN_UNREACHABLE: dict[str, str] = {}


@pytest.fixture(scope="module")
def c():
    return classify()


@pytest.fixture(scope="module")
def named():
    """Active question -> the terms its SPARQL names."""
    return {cid: terms_named(CQS[cid]["sparql"]) for cid in DEFAULT_ACTIVE}


def test_every_active_question_has_a_query():
    missing = [cid for cid in DEFAULT_ACTIVE if cid not in CQS]
    assert not missing, f"active questions with no SPARQL: {missing}"


def test_every_active_annotation_is_backed_by_its_query(c, named):
    """A term annotated with CQn must appear in CQn's SPARQL."""
    unbacked = [
        (t, cid)
        for t in sorted(c.retained)
        for cid in sorted(c.cqs(t))
        if cid in named and t not in named[cid]
        and t not in KNOWN_UNREACHABLE
    ]
    assert not unbacked, (
        f"{len(unbacked)} annotations name a query that never mentions the "
        f"term. Either the query is written more narrowly than its question "
        f"or the annotation is wrong:\n"
        + "\n".join(f"  {t} claims {cid}" for t, cid in unbacked))


def test_every_retained_term_is_reached_by_an_active_query(c, named):
    """The retention guarantee: no term survives on an unbacked label."""
    all_named = set().union(*named.values())
    unreached = sorted(t for t in c.retained
                       if t not in all_named and t not in KNOWN_UNREACHABLE)
    assert not unreached, (
        f"{len(unreached)} retained terms are named by no active query:\n  "
        + ", ".join(unreached))


def test_the_optional_only_group_is_reported_not_hidden(c):
    """Pins the number of retained terms named only inside an OPTIONAL.

    Such a term is traversed in the sense the criterion tests, but
    deleting it would cost the query a column rather than its rows. The
    count is pinned here, so a query rewrite that moves terms in or out
    of an OPTIONAL fails this test.
    """
    hard = set().union(*(terms_named(CQS[q]["sparql"])
                         - terms_named_optionally(CQS[q]["sparql"])
                         for q in DEFAULT_ACTIVE))
    opt_only = sorted(t for t in c.retained if t not in hard)
    assert len(opt_only) == 17, (
        f"{len(opt_only)} retained terms are named only inside an "
        f"OPTIONAL, not the 17 this test pins:\n  "
        + ", ".join(opt_only))


def test_no_query_names_an_undeclared_term(c, named):
    """Every kgp: term an active query names is declared in the ontology.

    Otherwise the criterion would be applied to a narrower vocabulary than
    the queries traverse.
    """
    declared = set(c.kinds)
    undeclared = sorted({t for terms in named.values() for t in terms}
                        - declared)
    assert not undeclared, (
        "active queries read kgp: terms the ontology does not declare: "
        + ", ".join(undeclared))
