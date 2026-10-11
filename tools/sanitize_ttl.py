#!/usr/bin/env python3
"""Percent-encode Turtle-invalid characters inside <...> IRIs.

rdflib's Turtle serializer is lax and can emit IRIs containing '['
']' '{' '}' '|' '^' '`' '"' ' ', which the Turtle grammar rejects, so
pyoxigraph refuses to parse them. This script rewrites
those characters as their percent-encoded equivalents so the same
graph loads under a strict parser. Only <...> IRIs without whitespace
are matched, so an IRI that contains a space is left unchanged.

Usage:
    python tools/sanitize_ttl.py input.ttl output.ttl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_BAD = set(' <>{}|^`"[]')

_IRI_PATTERN = re.compile(r"<([^\s>]*)>")


def _encode_iri(iri: str) -> str:
    out = []
    for ch in iri:
        if ch in _BAD or ord(ch) < 0x20:
            out.append("%%%02X" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _fix_line(line: str) -> tuple[str, int]:
    fixed = 0
    def _sub(m: re.Match) -> str:
        nonlocal fixed
        iri = m.group(1)
        if any(c in _BAD for c in iri):
            fixed += 1
            return "<" + _encode_iri(iri) + ">"
        return m.group(0)
    return _IRI_PATTERN.sub(_sub, line), fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    args = ap.parse_args()

    total = 0
    with open(args.inp, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            fixed_line, n = _fix_line(line)
            fout.write(fixed_line)
            total += n
    print(f"sanitized {total} IRIs -> {args.out}")


if __name__ == "__main__":
    main()
