#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build WORDxxxx -> Korean/gloss mapping from AIHub REAL/WORD morpheme JSON files.

Usage examples:
  python build_word_label_map.py --input "D:\\...\\01_real_word_morpheme.zip" --output .\\models\\label_map.json
  python build_word_label_map.py --input "D:\\...\\morpheme" --output .\\models\\label_map.json
"""

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

WORD_RE = re.compile(r"WORD(\d{4})", re.IGNORECASE)


def word_id_from_name(name: str) -> Optional[str]:
    m = WORD_RE.search(name)
    if not m:
        return None
    return f"WORD{m.group(1)}"


def iter_json_sources(input_path: Path) -> Iterable[Tuple[str, bytes]]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                yield info.filename, zf.read(info)
    elif input_path.is_dir():
        for p in input_path.rglob("*.json"):
            yield str(p), p.read_bytes()
    elif input_path.is_file() and input_path.suffix.lower() == ".json":
        yield str(input_path), input_path.read_bytes()
    else:
        raise FileNotFoundError(f"Unsupported input path: {input_path}")


def decode_json(raw: bytes) -> Optional[Any]:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    return None


def find_candidate_names(obj: Any) -> list:
    """Prefer AIHub morpheme attributes.name, but stay robust to schema variants."""
    names = []

    def walk(x: Any, parent_key: str = ""):
        if isinstance(x, dict):
            # Highest priority: attributes.name
            attrs = x.get("attributes")
            if isinstance(attrs, dict):
                n = attrs.get("name")
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())

            # Other common name-like keys
            for k in ("name", "word", "gloss", "label", "korean", "text"):
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    names.append(v.strip())

            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, parent_key)

    walk(obj)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="WORD morpheme zip, folder, or json file")
    ap.add_argument("--output", default="models/label_map.json")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    votes = defaultdict(Counter)
    scanned = 0
    json_ok = 0

    for name, raw in iter_json_sources(input_path):
        wid = word_id_from_name(name)
        if not wid:
            continue
        scanned += 1
        obj = decode_json(raw)
        if obj is None:
            continue
        json_ok += 1
        candidates = find_candidate_names(obj)
        # Remove obvious metadata-ish junk and WORD ids themselves
        clean = []
        for c in candidates:
            c = str(c).strip()
            if not c or WORD_RE.fullmatch(c):
                continue
            if c.lower().endswith((".jpg", ".mp4", ".json")):
                continue
            clean.append(c)
        for c in clean:
            votes[wid][c] += 1

    out: Dict[str, str] = {}
    debug_rows = {}
    for wid, counter in votes.items():
        if not counter:
            continue
        best, count = counter.most_common(1)[0]
        out[wid] = best
        if args.debug:
            debug_rows[wid] = counter.most_common(5)

    out = dict(sorted(out.items(), key=lambda kv: kv[0]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] scanned json files with WORD id: {scanned}")
    print(f"[DONE] parsed json files: {json_ok}")
    print(f"[DONE] mapping entries: {len(out)}")
    print(f"[DONE] saved: {out_path}")
    if args.debug:
        dbg = out_path.with_suffix(".debug.json")
        dbg.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] debug saved: {dbg}")


if __name__ == "__main__":
    main()
