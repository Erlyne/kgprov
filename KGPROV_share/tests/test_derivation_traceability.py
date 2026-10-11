"""The pruning classification and the documents that report it agree.

These tests pin the term counts and removal groups produced by
`prune_profile.classify()`, and, when the files are present, check the
derivation appendix (`out/appendix/app_derivation.tex`) and the thesis
source (`paper/thesis.tex`) against the same object. Neither file is in
this folder, so the four tests that read them skip.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.prune_profile import DEFAULT_ACTIVE, classify

ROOT = Path(__file__).resolve().parents[1]
APPENDIX = ROOT / "out" / "appendix" / "app_derivation.tex"
THESIS = ROOT / "paper" / "thesis.tex"


@pytest.fixture(scope="module")
def c():
    return classify()


# ---------------------------------------------------------------------
#  Term counts and removal groups
# ---------------------------------------------------------------------

def test_maximal_model_is_87_terms(c):
    """37 classes + 11 object properties + 39 datatype properties.

    Annotation properties describe the profile rather than belonging to
    it, and individuals are values of a class, not terms of their own, so
    neither is counted.
    """
    assert len(c.terms) == 87
    kinds = [c.kinds[t] for t in c.terms]
    assert kinds.count("class") == 37
    assert kinds.count("objprop") == 11
    assert kinds.count("dataprop") == 39


def test_criterion_retains_60_and_removes_27(c):
    assert len(c.retained) == 60
    assert len(c.removed) == 27
    assert (len(c.orphaned) + len(c.decided_out)
            + len(c.unannotated)) == 27


def test_the_withdrawn_question_strands_the_run_identity_vocabulary(c):
    """CQ14 is not in DEFAULT_ACTIVE, so the terms annotated only with
    CQ14 are orphaned."""
    assert "CQ14" not in DEFAULT_ACTIVE
    assert set(c.orphaned) == {
        "PipelineRun", "ConfigurationVariant", "modelId", "pipelineVersion",
        "configurationVariantLabel", "autonomyLevel",
        # CQ14 is DecodingConfiguration's only annotation; CQ11's SPARQL
        # does not name it.
        "DecodingConfiguration",
    }
    for t in c.orphaned:
        assert c.motiv[t] == {"CQ14"}, f"{t} is stranded by more than CQ14"


def test_decision_transfer_terms_are_decided_out(c):
    """The four decision-transfer terms in DECIDED_OUT are removed
    whatever their annotations say."""
    assert set(c.decided_out) == {
        "DecisionTransfer", "DeduplicationKey", "decidingContext",
        "decisionTransferredFrom",
    }


def test_narrowing_subclasses_are_removed_with_the_other_unannotated(c):
    """WindowScoping and BucketConstraintLookup carry no annotation.

    The criterion removes them like any other unannotated term, so every
    retained term is reached by some question.
    """
    for t in ("WindowScoping", "BucketConstraintLookup"):
        assert t in c.removed
        assert t in c.unannotated
        assert c.cqs(t) == []


def test_the_unannotated_are_exactly_what_the_chapter_lists(c):
    """The sixteen terms that carry no kgp:motivatedByCQ annotation.

    They include terms no run populates (probabilityMass,
    GeneratedRationale, rationaleText, ExpressibilityScope) and terms an
    active query cannot reach in the graph: NormalizedPredicate, whose
    kgp:appliesTo is a bare string rather than a link to its edge, and
    OntologySnapshot with ontologyVersion, which an attestation cannot
    reach because the authority state it points at is typed prov:Entity.
    """
    assert set(c.unannotated) == {
        "WindowScoping", "BucketConstraintLookup", "SourceManuscript",
        "resolvedString", "bioConditions",
        "ExpressibilityScope", "ExtractionAgent", "GeneratedRationale",
        "NormalizationAgent", "NormalizedPredicate", "OntologySnapshot",
        "RelationExtractionAgent", "TargetChunk", "ontologyVersion",
        "probabilityMass", "rationaleText",
    }


# ---------------------------------------------------------------------
#  The derivation appendix, when present, says the same thing
# ---------------------------------------------------------------------

@pytest.mark.skipif(not APPENDIX.exists(),
                    reason="appendix not built")
def test_appendix_summary_matches_the_classification(c):
    text = APPENDIX.read_text(encoding="utf-8")
    m = re.search(r"\\textbf\{Summary:\}\s*(\d+) terms,\s*(\d+) retained"
                  r".*?(\d+) removed", text, re.S)
    assert m, "appendix carries no summary line"
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (
        len(c.terms), len(c.retained), len(c.removed))


@pytest.mark.skipif(not APPENDIX.exists(), reason="appendix not built")
def test_appendix_never_lists_a_removed_term_as_retained(c):
    """No term the criterion removes appears in the retained block."""
    text = APPENDIX.read_text(encoding="utf-8")
    retained_block = text.split("Retained --- ")[1].split("Removed --- ")[0]
    for t in c.removed:
        assert f"\\texttt{{{t}}}" not in retained_block, \
            f"{t} is removed by the criterion but listed as retained"


@pytest.mark.skipif(not APPENDIX.exists(), reason="appendix not built")
def test_appendix_derivation_table_uses_only_active_questions(c):
    text = APPENDIX.read_text(encoding="utf-8")
    table = text.split("tab:app-derivation")[1].split(r"\end{longtable}")[0]
    rows = re.findall(r"^(CQ\d+) &", table, re.M)
    assert rows, "derivation table has no rows"
    assert set(rows) <= set(DEFAULT_ACTIVE), \
        f"derivation table cites withdrawn questions: " \
        f"{set(rows) - set(DEFAULT_ACTIVE)}"


# ---------------------------------------------------------------------
#  The thesis source, when present, prints the same numbers
# ---------------------------------------------------------------------

@pytest.mark.skipif(not THESIS.exists(), reason="thesis not present")
def test_chapter_six_prints_the_pruner_numbers(c):
    text = THESIS.read_text(encoding="utf-8")
    assert f"contains {len(c.terms)} terms" in text
    assert (rf"\textbf{{{len(c.retained)}}} and removes "
            rf"\textbf{{{len(c.removed)}}}") in text
