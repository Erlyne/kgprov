"""Loaders for BC5CDR (PubTator) and MeSH (Descriptor XML).

Two loaders, both pure parsing, no LLM calls, no side effects beyond
reading the files. Everything downstream is built on the invariants
established here: full_text = title + " " + abstract with offsets into
that concatenation, and one MeSH surface form per (descriptor, term).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from xml.etree import ElementTree as etree


# ---------------------------------------------------------------------
#  BC5CDR
# ---------------------------------------------------------------------

@dataclass
class GoldMention:
    start: int
    end: int
    text: str
    entity_type: str          # "Chemical" or "Disease"
    mesh_id: str | None       # None if "-1"
    is_compound: bool = False # True if raw MESH_ID had "|"


@dataclass
class GoldRelation:
    chemical_mesh_id: str
    disease_mesh_id: str


@dataclass
class Document:
    pmid: str
    title: str
    abstract: str
    full_text: str            # title + " " + abstract
    mentions: list[GoldMention] = field(default_factory=list)
    relations: list[GoldRelation] = field(default_factory=list)


def _split_mesh_id(raw: str) -> tuple[str | None, bool]:
    """Return (first_id_or_None, was_compound)."""
    raw = raw.strip()
    if raw == "-1" or raw == "":
        return None, False
    if "|" in raw:
        return raw.split("|", 1)[0], True
    return raw, False


def parse_pubtator_file(path: str | Path) -> list[Document]:
    """Parse one PubTator .txt file. Blank line separates documents."""
    path = Path(path)
    docs: list[Document] = []
    current_pmid: str | None = None
    title = ""
    abstract = ""
    mentions: list[GoldMention] = []
    relations: list[GoldRelation] = []

    def flush() -> None:
        nonlocal title, abstract, mentions, relations, current_pmid
        if current_pmid is None:
            return
        full_text = title + " " + abstract if abstract else title
        docs.append(Document(
            pmid=current_pmid, title=title, abstract=abstract,
            full_text=full_text, mentions=mentions, relations=relations))
        current_pmid = None
        title = ""
        abstract = ""
        mentions = []
        relations = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                flush()
                continue

            # Title / abstract lines use "|" separators.
            # Split on the first two "|" only, since text may contain "|".
            if "|t|" in line or "|a|" in line:
                pmid, kind, text = line.split("|", 2)
                if current_pmid is None:
                    current_pmid = pmid
                if kind == "t":
                    title = text
                elif kind == "a":
                    abstract = text
                continue

            # Annotation lines use TAB separators.
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            pmid = parts[0]
            if current_pmid is None:
                current_pmid = pmid

            # CID relation: PMID \t CID \t CHEM_ID \t DIS_ID
            if len(parts) >= 4 and parts[1] == "CID":
                relations.append(GoldRelation(
                    chemical_mesh_id=parts[2], disease_mesh_id=parts[3]))
                continue

            # Entity mention: PMID \t START \t END \t TEXT \t TYPE \t MESH_ID
            if len(parts) >= 6:
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                text = parts[3]
                etype = parts[4]
                mesh_raw = parts[5]
                mid, compound = _split_mesh_id(mesh_raw)
                mentions.append(GoldMention(
                    start=start, end=end, text=text,
                    entity_type=etype, mesh_id=mid, is_compound=compound))

        flush()

    return docs


def load_bc5cdr(bc5cdr_dir: str | Path, split: str = "all") -> list[Document]:
    """Load one or more PubTator files. split in {train, dev, test, all}."""
    d = Path(bc5cdr_dir)
    files = {
        "train": "CDR_TrainingSet.PubTator.txt",
        "dev":   "CDR_DevelopmentSet.PubTator.txt",
        "test":  "CDR_TestSet.PubTator.txt",
    }
    if split == "all":
        selected = list(files.values())
    elif split in files:
        selected = [files[split]]
    else:
        raise ValueError(f"unknown split: {split}")
    docs: list[Document] = []
    for name in selected:
        docs.extend(parse_pubtator_file(d / name))
    return docs


def bc5cdr_summary(docs: list[Document]) -> dict:
    n_docs = len(docs)
    n_mentions = 0
    n_with_gold = 0
    n_no_gold = 0
    n_compound = 0
    n_chemical = 0
    n_disease = 0
    n_cid = 0
    for d in docs:
        n_mentions += len(d.mentions)
        for m in d.mentions:
            if m.mesh_id is None:
                n_no_gold += 1
            else:
                n_with_gold += 1
            if m.is_compound:
                n_compound += 1
            if m.entity_type == "Chemical":
                n_chemical += 1
            elif m.entity_type == "Disease":
                n_disease += 1
        n_cid += len(d.relations)
    return {
        "documents": n_docs,
        "gold_mentions": n_mentions,
        "mentions_with_mesh_id": n_with_gold,
        "mentions_no_gold": n_no_gold,
        "compound_mesh_ids": n_compound,
        "chemical_mentions": n_chemical,
        "disease_mentions": n_disease,
        "cid_relations": n_cid,
    }


# ---------------------------------------------------------------------
#  MeSH
# ---------------------------------------------------------------------

@dataclass
class OntologyEntry:
    descriptor_ui: str
    preferred_name: str
    synonyms: list[str]
    tree_numbers: list[str]
    entity_types: set[str]    # {"Chemical"} or {"Disease"} or both, or empty


@dataclass
class MeshOntology:
    version: str
    entries: dict[str, OntologyEntry]        # ui -> entry (all descriptors)
    surface_forms: list[tuple[str, str]]     # (surface, ui) for BC5CDR-compatible only

    def contains(self, ui: str) -> bool:
        return ui in self.entries

    def compatible(self, ui: str, entity_type: str) -> bool:
        e = self.entries.get(ui)
        if e is None:
            return False
        return entity_type in e.entity_types


def _types_from_trees(tree_numbers: list[str]) -> set[str]:
    """Tree prefix C -> Disease, prefix D -> Chemical.
    Descriptors with trees under BOTH categories are compatible with either."""
    types: set[str] = set()
    for t in tree_numbers:
        if not t:
            continue
        prefix = t[0]
        if prefix == "C":
            types.add("Disease")
        elif prefix == "D":
            types.add("Chemical")
    return types


def parse_mesh_xml(path: str | Path, version: str) -> MeshOntology:
    """Streaming parse of desc<year>.xml. Memory-safe at 300MB."""
    path = Path(path)
    entries: dict[str, OntologyEntry] = {}
    surface_forms: list[tuple[str, str]] = []

    context = etree.iterparse(str(path), events=("end",))

    for _, rec in context:
        if rec.tag != "DescriptorRecord":
            continue
        ui_el = rec.find("DescriptorUI")
        name_el = rec.find("DescriptorName/String")
        if ui_el is None or name_el is None:
            rec.clear()
            continue
        ui = ui_el.text or ""
        pref = name_el.text or ""

        tree_numbers: list[str] = []
        for tn in rec.iterfind("TreeNumberList/TreeNumber"):
            if tn.text:
                tree_numbers.append(tn.text)

        # Collect all Term/String across all Concepts
        term_strings: list[str] = []
        for t in rec.iterfind("ConceptList/Concept/TermList/Term/String"):
            if t.text:
                term_strings.append(t.text)

        # Preferred name may or may not appear in TermList; dedupe with
        # preferred name always present.
        seen: set[str] = set()
        synonyms: list[str] = []
        for s in [pref] + term_strings:
            if s and s not in seen:
                seen.add(s)
                synonyms.append(s)

        types = _types_from_trees(tree_numbers)
        entry = OntologyEntry(
            descriptor_ui=ui, preferred_name=pref, synonyms=synonyms,
            tree_numbers=tree_numbers, entity_types=types)
        entries[ui] = entry

        # Surface form table: BC5CDR-compatible descriptors only.
        if types & {"Chemical", "Disease"}:
            for s in synonyms:
                surface_forms.append((s, ui))

        # Free the element to keep memory bounded (stdlib pattern).
        rec.clear()

    return MeshOntology(version=version, entries=entries,
                        surface_forms=surface_forms)


def mesh_summary(onto: MeshOntology) -> dict:
    n_chem = sum(1 for e in onto.entries.values() if "Chemical" in e.entity_types)
    n_dis = sum(1 for e in onto.entries.values() if "Disease" in e.entity_types)
    n_both = sum(1 for e in onto.entries.values()
                 if "Chemical" in e.entity_types and "Disease" in e.entity_types)
    n_neither = sum(1 for e in onto.entries.values() if not e.entity_types)
    return {
        "version": onto.version,
        "descriptors_total": len(onto.entries),
        "descriptors_chemical": n_chem,
        "descriptors_disease": n_dis,
        "descriptors_both_C_and_D": n_both,
        "descriptors_neither_C_nor_D": n_neither,
        "surface_forms": len(onto.surface_forms),
    }


def mesh_diff(onto_old: MeshOntology, onto_new: MeshOntology) -> dict:
    """Diff two MeSH releases. Retired = present old, absent new."""
    old = onto_old.entries
    new = onto_new.entries
    retired = sorted(ui for ui in old if ui not in new)
    name_changed = {}
    for ui, e_old in old.items():
        e_new = new.get(ui)
        if e_new is not None and e_new.preferred_name != e_old.preferred_name:
            name_changed[ui] = {
                onto_old.version: e_old.preferred_name,
                onto_new.version: e_new.preferred_name,
            }
    return {
        "retired": retired,
        "name_changed": name_changed,
        f"n_{onto_old.version}": len(old),
        f"n_{onto_new.version}": len(new),
        "n_retired": len(retired),
        "n_name_changed": len(name_changed),
    }


# ---------------------------------------------------------------------
#  Convenience: verify offsets on document 1
# ---------------------------------------------------------------------

def verify_offsets(docs: list[Document]) -> list[dict]:
    """Return list of mismatches (empty on success). Raises nothing."""
    mismatches = []
    for d in docs:
        for m in d.mentions:
            got = d.full_text[m.start:m.end]
            if got != m.text:
                mismatches.append({
                    "pmid": d.pmid, "start": m.start, "end": m.end,
                    "expected": m.text, "got": got})
    return mismatches


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Load BC5CDR + MeSH and print summary")
    ap.add_argument("--bc5cdr-dir", default="data/bc5cdr")
    ap.add_argument("--split", default="all", choices=["train", "dev", "test", "all"])
    ap.add_argument("--mesh-2023", default="data/mesh/desc2023.xml")
    ap.add_argument("--mesh-2025", default="data/mesh/desc2025.xml")
    ap.add_argument("--diff-out", default="out/results/mesh_diff.json")
    ap.add_argument("--no-mesh", action="store_true",
                    help="Skip MeSH parsing (fast smoke test)")
    ap.add_argument("--verify-offsets", action="store_true")
    args = ap.parse_args()

    print(f"[bc5cdr] loading split={args.split} from {args.bc5cdr_dir}")
    docs = load_bc5cdr(args.bc5cdr_dir, args.split)
    bc = bc5cdr_summary(docs)
    print(f"[bc5cdr] summary: {json.dumps(bc, indent=2)}")

    if args.verify_offsets or True:
        mm = verify_offsets(docs[:1])
        if mm:
            print(f"[bc5cdr] OFFSET MISMATCH on doc 1: {mm[:3]}")
        else:
            print(f"[bc5cdr] offsets on doc 1 OK ({len(docs[0].mentions)} mentions)")

    if args.no_mesh:
        return

    print(f"[mesh] parsing {args.mesh_2023}")
    o23 = parse_mesh_xml(args.mesh_2023, "2023")
    print(f"[mesh] 2023 summary: {json.dumps(mesh_summary(o23), indent=2)}")

    print(f"[mesh] parsing {args.mesh_2025}")
    o25 = parse_mesh_xml(args.mesh_2025, "2025")
    print(f"[mesh] 2025 summary: {json.dumps(mesh_summary(o25), indent=2)}")

    diff = mesh_diff(o23, o25)
    Path(args.diff_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.diff_out, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2)
    print(f"[mesh] wrote diff to {args.diff_out}: "
          f"{diff['n_retired']} retired, {diff['n_name_changed']} name-changed")

    # How many retired IDs are gold answers in the corpus?
    gold_ids = {m.mesh_id for d in docs for m in d.mentions if m.mesh_id}
    retired_that_are_gold = sorted(gold_ids & set(diff["retired"]))
    print(f"[mesh] retired IDs that are BC5CDR gold answers: "
          f"{len(retired_that_are_gold)} / {len(gold_ids)} total gold IDs")

    # Also: gold IDs absent from the 2025 ontology entirely
    absent_from_2025 = sorted(gold_ids - set(o25.entries.keys()))
    print(f"[mesh] gold IDs absent from desc2025 entirely: {len(absent_from_2025)}")


if __name__ == "__main__":
    _main()
