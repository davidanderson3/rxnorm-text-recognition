#!/usr/bin/env python3
"""RxNorm text recognition: build an index from RRF files and map free text to target TTYs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

TARGET_TTYS: Tuple[str, ...] = (
    "SBD",
    "SCD",
    "GPCK",
    "BPCK",
    "BN",
    "SCDC",
    "IN",
    "PIN",
    "MIN",
)

TTY_PRIORITY: Dict[str, int] = {tty: i for i, tty in enumerate(TARGET_TTYS)}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[+/%_-][A-Za-z0-9]+)*(?:\.[A-Za-z0-9]+)*")
STRENGTH_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|meq|%)\b"
    r"(?!\s*(?:/|per)\s*(?:dl|l|ml|kg|m2|hr|h)\b)"
)
RATIO_STRENGTH_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:[-/]|to|\s+)\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|meq|%)\b"
)
SINGLE_STRENGTH_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|meq|%)\b")
SLASH_RATIO_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b")
BARE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
XML_VALUE_ATTR_RE = re.compile(r"""value\s*=\s*(['"])(.*?)\1""", re.IGNORECASE)

CONFUSABLE_CHAR_MAP: Dict[str, str] = {
    "∕": "/",
    "／": "/",
    "⁄": "/",
    "÷": "/",
    "ø": "o",
    "Ø": "O",
    "ð": "d",
    "Ð": "D",
    "ł": "l",
    "Ł": "L",
    "υ": "u",
    "Υ": "U",
    "α": "a",
    "Α": "A",
    "β": "b",
    "Β": "B",
    "ο": "o",
    "Ο": "O",
    "ρ": "p",
    "Ρ": "P",
    "е": "e",
    "Е": "E",
    "а": "a",
    "А": "A",
    "с": "c",
    "С": "C",
    "р": "p",
    "Р": "P",
    "х": "x",
    "Х": "X",
    "к": "k",
    "К": "K",
}

CONTEXT_STOPWORDS: Set[str] = {
    "po",
    "oral",
    "iv",
    "im",
    "sq",
    "sc",
    "subcutaneous",
    "daily",
    "nightly",
    "bid",
    "tid",
    "qid",
    "qd",
    "qhs",
    "prn",
    "qod",
    "continue",
    "med",
    "start",
    "meds",
    "medications",
    "take",
    "takes",
    "tablet",
    "capsule",
    "tab",
    "cap",
    "pill",
    "and",
    "with",
    "for",
    "x7d",
    "x10d",
    "hfa",
    "mdi",
    "inh",
    "inhaler",
    "spray",
    "aerosol",
    "er",
    "xr",
    "ec",
}

NOISY_TOKEN_REPLACEMENTS: Dict[str, str] = {
    "losrtan": "losartan",
    "liptor": "lipitor",
    "atorvstatin": "atorvastatin",
    "atorvststin": "atorvastatin",
    "carvadiol": "carvedilol",
    "furosemid": "furosemide",
    "albterol": "albuterol",
    "albutrol": "albuterol",
    "amxclv": "amoxicillin clavulanate",
    "amoxclv": "amoxicillin clavulanate",
    "amoxclav": "amoxicillin clavulanate",
    "hctz": "hydrochlorothiazide",
    "asa": "aspirin",
    "apap": "acetaminophen",
    "oxyapap": "acetaminophen oxycodone",
    "pred": "prednisone",
    "traz": "trazodone",
    "trazadone": "trazodone",
    "glipzide": "glipizide",
    "inslin": "insulin",
    "duoneb": "albuterol ipratropium",
    "norepi": "norepinephrine",
    "levophed": "norepinephrine",
    "dex": "dexmedetomidine",
    "vanc": "vancomycin",
    "zosyn": "piperacillin tazobactam",
}

FORM_HINTS: Dict[str, Tuple[str, ...]] = {
    "inhaler": ("inhaler", "hfa", "mdi", "actuat", "metered dose", "aerosol", "spray"),
    "dry_powder": ("dry powder", "diskus"),
    "tablet": ("tablet", "tab", "oral tablet"),
    "capsule": ("capsule", "cap"),
    "solution": ("solution", "soln"),
    "suspension": ("suspension",),
    "injection": (
        "inject",
        "injection",
        "intravenous",
        "subcutaneous",
        "intramuscular",
        "iv",
        "ivp",
        "ivpb",
        "gtt",
        "drip",
        "infusion",
    ),
    "dr": ("delayed release", "dr"),
    "xl24": ("xl", "24 hr"),
    "cream": ("cream",),
    "ointment": ("ointment",),
    "patch": ("patch", "transdermal"),
    "sublingual": ("sublingual",),
    "nasal": ("nasal",),
    "ophthalmic": ("ophthalmic", "eye"),
    "otologic": ("otic", "ear"),
    "er": ("extended release", "er", "xr", "xl", "sr", "24 hr", "12 hr"),
    "ec": ("enteric", "ec"),
}

NON_DRUG_EXACT_TERMS: Set[str] = {
    "pill",
    "sugar pill",
    "cholesterol",
    "lactate",
    "water",
}

DENSE_LINE_MED_TOKENS: Dict[str, str] = {
    "dexmedetomidine": "dexmedetomidine",
    "metoprolol": "metoprolol",
    "lisinopril": "lisinopril",
    "metformin": "metformin",
    "glyburide": "glyburide",
    "atorvastatin": "atorvastatin",
    "rosuvastatin": "rosuvastatin",
    "propofol": "propofol",
    "fentanyl": "fentanyl",
    "prednisone": "prednisone",
    "heparin": "heparin",
    "eliquis": "eliquis",
    "apixaban": "apixaban",
    "vanc": "vancomycin",
    "vancomycin": "vancomycin",
    "zosyn": "zosyn",
    "norepi": "norepinephrine",
    "dex": "dexmedetomidine",
}

DENSE_LINE_MED_RE = re.compile(
    "|".join(
        re.escape(token) for token in sorted(DENSE_LINE_MED_TOKENS.keys(), key=len, reverse=True)
    ),
    flags=re.IGNORECASE,
)

NON_DRUG_EXACT_TTYS: Set[str] = {
    "DF",
    "DFG",
    "SCDG",
    "SBDG",
    "SCDF",
    "SBDF",
}

STRENGTH_UNIT_TOKENS: Set[str] = {
    "mg",
    "mcg",
    "g",
    "ml",
    "unit",
    "units",
    "meq",
    "pct",
    "percent",
}

SEGMENT_SPLIT_PATTERNS: Tuple[str, ...] = (
    r"\?",
    r"!",
    r";",
    r":",
    r"\+",
    r"\bbut\b",
    r"\bplus\b",
)

IV_ROUTE_TOKENS: Set[str] = {"iv", "ivp", "ivpb", "intravenous"}
ORAL_ROUTE_TOKENS: Set[str] = {"po", "oral"}

DIRECT_INGREDIENT_RELAS: Set[str] = {
    "has_ingredient",
    "ingredient_of",
    "has_precise_ingredient",
    "precise_ingredient_of",
    "has_form",
    "form_of",
    "has_part",
    "part_of",
    "has_ingredients",
    "ingredients_of",
}

INGREDIENT_BRIDGE_RELAS: Set[str] = {
    "consists_of",
    "constitutes",
    "has_tradename",
    "tradename_of",
    "contains",
    "contained_in",
    "has_ingredient",
    "ingredient_of",
    "has_form",
    "form_of",
}

TARGET_RELAS: Dict[str, Set[str]] = {
    "SBD": {
        "consists_of",
        "constitutes",
        "has_tradename",
        "tradename_of",
        "has_ingredient",
        "ingredient_of",
        "has_ingredients",
        "ingredients_of",
        "has_part",
        "part_of",
        "contains",
        "contained_in",
        "isa",
        "inverse_isa",
        "has_form",
        "form_of",
        "has_quantified_form",
        "quantified_form_of",
    },
    "SCD": {
        "consists_of",
        "constitutes",
        "has_tradename",
        "tradename_of",
        "has_ingredient",
        "ingredient_of",
        "has_ingredients",
        "ingredients_of",
        "has_part",
        "part_of",
        "contains",
        "contained_in",
        "isa",
        "inverse_isa",
        "has_form",
        "form_of",
        "has_quantified_form",
        "quantified_form_of",
    },
    "GPCK": {
        "contains",
        "contained_in",
        "has_part",
        "part_of",
        "has_ingredients",
        "ingredients_of",
        "consists_of",
        "constitutes",
    },
    "BPCK": {
        "contains",
        "contained_in",
        "has_part",
        "part_of",
        "has_ingredients",
        "ingredients_of",
        "consists_of",
        "constitutes",
    },
    "BN": {
        "has_tradename",
        "tradename_of",
        "has_ingredient",
        "ingredient_of",
        "consists_of",
        "constitutes",
    },
    "SCDC": {
        "has_ingredient",
        "ingredient_of",
        "consists_of",
        "constitutes",
        "has_form",
        "form_of",
        "has_precise_ingredient",
        "precise_ingredient_of",
    },
    "IN": {
        "has_ingredient",
        "ingredient_of",
        "has_tradename",
        "tradename_of",
        "consists_of",
        "constitutes",
        "has_form",
        "form_of",
        "has_precise_ingredient",
        "precise_ingredient_of",
        "has_part",
        "part_of",
    },
    "PIN": {
        "has_form",
        "form_of",
        "has_tradename",
        "tradename_of",
        "consists_of",
        "constitutes",
        "has_precise_ingredient",
        "precise_ingredient_of",
        "has_ingredient",
        "ingredient_of",
    },
    "MIN": {
        "has_part",
        "part_of",
        "has_ingredients",
        "ingredients_of",
        "has_ingredient",
        "ingredient_of",
        "consists_of",
        "constitutes",
        "contains",
        "contained_in",
    },
}


def log(message: str) -> None:
    print(message, file=sys.stderr)


def normalize_text(value: str) -> str:
    if value:
        value = "".join(CONFUSABLE_CHAR_MAP.get(ch, ch) for ch in value)
    ascii_text = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    lowered = ascii_text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def hash_embedding(text: str, dims: int, min_n: int = 3, max_n: int = 5) -> np.ndarray:
    vec = np.zeros(dims, dtype=np.float32)
    if not text:
        return vec

    padded = f" {text} "
    for n in range(min_n, max_n + 1):
        if len(padded) < n:
            continue
        for i in range(0, len(padded) - n + 1):
            gram = padded[i : i + n]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            hashed = int.from_bytes(digest, byteorder="little", signed=False)
            idx = hashed % dims
            sign = 1.0 if (hashed & (1 << 63)) == 0 else -1.0
            vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


@dataclass
class ConceptAccumulator:
    preferred_name: str = ""
    preferred_rank: Tuple[int, int, int] = (9, 999, 999999)
    tty_name_rank: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    tty_names: Dict[str, str] = field(default_factory=dict)
    alias_terms: List[str] = field(default_factory=list)

    def add(
        self,
        tty: str,
        raw_name: str,
        norm_name: str,
        is_pref: str,
        max_aliases_per_concept: int,
    ) -> None:
        pref_rank = 0 if is_pref == "Y" else 1
        tty_rank = TTY_PRIORITY.get(tty, len(TARGET_TTYS))
        rank = (pref_rank, tty_rank, len(raw_name))

        if rank < self.preferred_rank:
            self.preferred_rank = rank
            self.preferred_name = raw_name

        current_tty_rank = self.tty_name_rank.get(tty)
        if current_tty_rank is None or rank < current_tty_rank:
            self.tty_name_rank[tty] = rank
            self.tty_names[tty] = raw_name

        if (
            norm_name
            and len(self.alias_terms) < max_aliases_per_concept
            and norm_name not in self.alias_terms
        ):
            self.alias_terms.append(norm_name)


def ensure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        DROP TABLE IF EXISTS concepts;
        DROP TABLE IF EXISTS concept_tty;
        DROP TABLE IF EXISTS alias;
        DROP TABLE IF EXISTS edges;

        CREATE TABLE concepts (
            rxcui TEXT PRIMARY KEY,
            preferred_name TEXT NOT NULL
        );

        CREATE TABLE concept_tty (
            rxcui TEXT NOT NULL,
            tty TEXT NOT NULL,
            name TEXT NOT NULL,
            PRIMARY KEY (rxcui, tty)
        );

        CREATE TABLE alias (
            norm_text TEXT NOT NULL,
            rxcui TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            tty TEXT NOT NULL
        );

        CREATE TABLE edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            rel TEXT,
            rela TEXT
        );
        """
    )
    conn.commit()


def load_rxnconso(
    conso_path: Path,
    conn: sqlite3.Connection,
    max_aliases_per_concept: int,
    max_concepts: Optional[int],
) -> Dict[str, ConceptAccumulator]:
    concepts: Dict[str, ConceptAccumulator] = {}
    alias_rows: List[Tuple[str, str, str, str]] = []

    kept_rows = 0
    line_count = 0
    with conso_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line_count += 1
            if line_count % 500000 == 0:
                log(f"[build-index] RXNCONSO lines scanned: {line_count:,}")

            cols = line.rstrip("\n").split("|")
            if len(cols) < 18:
                continue

            rxcui = cols[0]
            lat = cols[1]
            ispref = cols[6]
            sab = cols[11]
            tty = cols[12]
            raw_name = cols[14]
            suppress = cols[16]

            if sab != "RXNORM" or lat != "ENG":
                continue
            if suppress in {"Y", "O"}:
                continue

            norm_name = normalize_text(raw_name)
            if not norm_name:
                continue

            accumulator = concepts.setdefault(rxcui, ConceptAccumulator())
            accumulator.add(
                tty=tty,
                raw_name=raw_name,
                norm_name=norm_name,
                is_pref=ispref,
                max_aliases_per_concept=max_aliases_per_concept,
            )

            alias_rows.append((norm_name, rxcui, raw_name, tty))
            if len(alias_rows) >= 15000:
                conn.executemany(
                    "INSERT INTO alias(norm_text, rxcui, raw_text, tty) VALUES (?, ?, ?, ?)",
                    alias_rows,
                )
                conn.commit()
                alias_rows.clear()

            kept_rows += 1
            if max_concepts is not None and len(concepts) >= max_concepts:
                break

    if alias_rows:
        conn.executemany(
            "INSERT INTO alias(norm_text, rxcui, raw_text, tty) VALUES (?, ?, ?, ?)",
            alias_rows,
        )
        conn.commit()

    log(
        f"[build-index] RXNCONSO kept rows: {kept_rows:,}, unique concepts: {len(concepts):,}"
    )
    return concepts


def persist_concepts(
    conn: sqlite3.Connection, concepts: Dict[str, ConceptAccumulator]
) -> List[str]:
    concept_rows: List[Tuple[str, str]] = []
    concept_tty_rows: List[Tuple[str, str, str]] = []
    rxcui_order: List[str] = sorted(concepts.keys())

    for rxcui in rxcui_order:
        item = concepts[rxcui]
        preferred = item.preferred_name or (item.alias_terms[0] if item.alias_terms else rxcui)
        concept_rows.append((rxcui, preferred))
        for tty, name in item.tty_names.items():
            concept_tty_rows.append((rxcui, tty, name))

    conn.executemany(
        "INSERT INTO concepts(rxcui, preferred_name) VALUES (?, ?)",
        concept_rows,
    )
    conn.executemany(
        "INSERT INTO concept_tty(rxcui, tty, name) VALUES (?, ?, ?)",
        concept_tty_rows,
    )
    conn.commit()
    return rxcui_order


def load_rxnrel(
    rel_path: Path,
    conn: sqlite3.Connection,
    known_rxcuis: Set[str],
    max_rel_lines: Optional[int],
) -> int:
    edge_rows: List[Tuple[str, str, str, str]] = []
    edge_count = 0
    line_count = 0
    with rel_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line_count += 1
            if line_count % 500000 == 0:
                log(f"[build-index] RXNREL lines scanned: {line_count:,}")

            cols = line.rstrip("\n").split("|")
            if len(cols) < 16:
                continue

            src = cols[0]
            rel = cols[3]
            dst = cols[4]
            rela = cols[7]
            sab = cols[10]
            suppress = cols[14]

            if sab != "RXNORM":
                continue
            if suppress in {"Y", "O"}:
                continue
            if not src or not dst or src == dst:
                continue
            if src not in known_rxcuis or dst not in known_rxcuis:
                continue

            edge_rows.append((src, dst, rel, rela))
            edge_rows.append((dst, src, rel, rela))
            edge_count += 2

            if len(edge_rows) >= 25000:
                conn.executemany(
                    "INSERT INTO edges(src, dst, rel, rela) VALUES (?, ?, ?, ?)",
                    edge_rows,
                )
                conn.commit()
                edge_rows.clear()

            if max_rel_lines is not None and line_count >= max_rel_lines:
                break

    if edge_rows:
        conn.executemany(
            "INSERT INTO edges(src, dst, rel, rela) VALUES (?, ?, ?, ?)",
            edge_rows,
        )
        conn.commit()

    log(f"[build-index] RXNREL kept edges (directed): {edge_count:,}")
    return edge_count


def build_embeddings(
    concepts: Dict[str, ConceptAccumulator], rxcui_order: Sequence[str], dims: int
) -> np.ndarray:
    vectors = np.zeros((len(rxcui_order), dims), dtype=np.float32)
    for idx, rxcui in enumerate(rxcui_order):
        item = concepts[rxcui]
        text_for_embedding = " ".join(item.alias_terms)
        if not text_for_embedding:
            text_for_embedding = normalize_text(item.preferred_name)
        vectors[idx] = hash_embedding(text_for_embedding, dims=dims)
        if (idx + 1) % 20000 == 0:
            log(f"[build-index] concept embeddings built: {idx + 1:,}")
    return vectors


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_alias_norm_text ON alias(norm_text);
        CREATE INDEX idx_edges_src ON edges(src);
        CREATE INDEX idx_concept_tty_rxcui ON concept_tty(rxcui);
        """
    )
    conn.commit()


def cmd_build_index(args: argparse.Namespace) -> int:
    rrf_dir = Path(args.rrf_dir).expanduser().resolve()
    conso_path = rrf_dir / "RXNCONSO.RRF"
    rel_path = rrf_dir / "RXNREL.RRF"
    if not conso_path.exists():
        raise FileNotFoundError(f"Missing RXNCONSO: {conso_path}")
    if not rel_path.exists():
        raise FileNotFoundError(f"Missing RXNREL: {rel_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_parent(out_dir)
    db_path = out_dir / "rxnorm_index.sqlite"
    emb_path = out_dir / "concept_embeddings.npy"
    rxcui_path = out_dir / "concept_rxcuis.json"
    meta_path = out_dir / "metadata.json"

    log(f"[build-index] writing index to: {out_dir}")
    conn = sqlite3.connect(str(db_path))
    try:
        init_db(conn)
        concepts = load_rxnconso(
            conso_path=conso_path,
            conn=conn,
            max_aliases_per_concept=args.max_aliases_per_concept,
            max_concepts=args.max_concepts,
        )
        rxcui_order = persist_concepts(conn, concepts)
        edge_count = load_rxnrel(
            rel_path=rel_path,
            conn=conn,
            known_rxcuis=set(rxcui_order),
            max_rel_lines=args.max_rel_lines,
        )
        create_indexes(conn)

        log("[build-index] building concept embeddings")
        vectors = build_embeddings(concepts, rxcui_order, dims=args.dims)
        np.save(emb_path, vectors)

        with rxcui_path.open("w", encoding="utf-8") as handle:
            json.dump(rxcui_order, handle)

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rrf_dir": str(rrf_dir),
            "concept_count": len(rxcui_order),
            "edge_count_directed": edge_count,
            "embedding_dims": args.dims,
            "max_aliases_per_concept": args.max_aliases_per_concept,
            "target_ttys": list(TARGET_TTYS),
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        log("[build-index] done")
        log(f"[build-index] sqlite: {db_path}")
        log(f"[build-index] embeddings: {emb_path}")
        return 0
    finally:
        conn.close()


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def detect_mentions(
    text: str,
    conn: sqlite3.Connection,
    max_ngram: int,
    max_exact_candidates: int,
) -> List[Dict[str, object]]:
    tokens = tokenize_with_spans(text)
    mentions: List[Dict[str, object]] = []
    if not tokens:
        stripped = text.strip()
        return (
            [
                {
                    "text": stripped,
                    "norm_text": normalize_text(stripped),
                    "start": 0,
                    "end": len(stripped),
                    "exact_rxcuids": [],
                }
            ]
            if stripped
            else []
        )

    occupied = [False] * len(tokens)
    max_n = min(max_ngram, len(tokens))

    for n in range(max_n, 0, -1):
        i = 0
        while i <= len(tokens) - n:
            if any(occupied[i : i + n]):
                i += 1
                continue

            start = tokens[i][1]
            end = tokens[i + n - 1][2]
            span_text = text[start:end]
            if "\n" in span_text:
                i += 1
                continue
            if n > 1 and not allow_multi_token_span(span_text):
                i += 1
                continue
            norm_span = normalize_text(span_text)
            if not norm_span:
                i += 1
                continue

            if norm_span in NON_DRUG_EXACT_TERMS:
                i += 1
                continue

            lookup_norms: List[str] = [norm_span]
            noisy_norm = normalize_noisy_text(span_text)
            if noisy_norm and noisy_norm not in lookup_norms:
                lookup_norms.append(noisy_norm)
            stripped_noisy = strip_context_tokens(noisy_norm)
            if stripped_noisy and stripped_noisy not in lookup_norms:
                lookup_norms.append(stripped_noisy)
            for existing in list(lookup_norms):
                alpha_tokens = [
                    tok
                    for tok in existing.split()
                    if any(ch.isalpha() for ch in tok)
                    and tok not in STRENGTH_UNIT_TOKENS
                    and tok not in CONTEXT_STOPWORDS
                ]
                if alpha_tokens:
                    alpha_only = " ".join(alpha_tokens)
                    if alpha_only and alpha_only not in lookup_norms:
                        lookup_norms.append(alpha_only)
                if 2 <= len(alpha_tokens) <= 3:
                    alpha_sorted = " ".join(sorted(alpha_tokens))
                    if alpha_sorted and alpha_sorted not in lookup_norms:
                        lookup_norms.append(alpha_sorted)

            rows: List[Tuple[str, str]] = []
            matched_norm = norm_span
            for lookup_norm in lookup_norms:
                candidate_rows = lookup_exact_alias_rows(conn, lookup_norm, max_exact_candidates)
                if not candidate_rows:
                    continue
                rows = candidate_rows
                matched_norm = lookup_norm
                break
            if not rows:
                i += 1
                continue

            for j in range(i, i + n):
                occupied[j] = True

            mentions.append(
                {
                    "text": span_text,
                    "norm_text": matched_norm,
                    "start": start,
                    "end": end,
                    "exact_rxcuids": [row[0] for row in rows],
                    "exact_ttys": sorted({row[1] for row in rows if row[1]}),
                }
            )
            i += n

    for idx, (token_text, start, end) in enumerate(tokens):
        if occupied[idx]:
            continue
        token_norm = normalize_noisy_text(token_text)
        replacement = NOISY_TOKEN_REPLACEMENTS.get(token_norm)
        if not replacement:
            continue
        replacement_norm = normalize_noisy_text(replacement)
        rows = lookup_exact_alias_rows(conn, replacement_norm, max_exact_candidates)
        mentions.append(
            {
                "text": token_text,
                "norm_text": replacement_norm,
                "start": start,
                "end": end,
                "exact_rxcuids": [row[0] for row in rows],
                "exact_ttys": sorted({row[1] for row in rows if row[1]}),
            }
        )
        occupied[idx] = True

    if text:
        line_mention_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for mention in mentions:
            bounds = get_line_bounds(text, int(mention["start"]), int(mention["end"]))
            line_mention_counts[bounds] += 1

        existing_spans: Set[Tuple[int, int]] = {
            (int(mention["start"]), int(mention["end"])) for mention in mentions
        }
        for line_start, line_end, line_text in iter_line_spans(text):
            if not should_scan_dense_line(line_text):
                continue
            if line_mention_counts.get((line_start, line_end), 0) >= 2:
                continue
            dense_mentions = extract_dense_line_mentions(
                line_text=line_text,
                line_start=line_start,
                conn=conn,
                max_exact_candidates=max_exact_candidates,
            )
            for dense in dense_mentions:
                dense_span = (int(dense["start"]), int(dense["end"]))
                if dense_span in existing_spans:
                    continue
                mentions.append(dense)
                existing_spans.add(dense_span)

    mentions.sort(key=lambda item: int(item["start"]))
    if mentions:
        line_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        line_bounds: List[Tuple[int, int]] = []
        for idx, mention in enumerate(mentions):
            span_start = int(mention["start"])
            span_end = int(mention["end"])
            bounds = get_line_bounds(text, span_start, span_end)
            line_bounds.append(bounds)
            line_to_indices[bounds].append(idx)

        drop_indices: Set[int] = set()
        for _, indices in line_to_indices.items():
            if len(indices) < 2:
                continue
            for i in indices:
                span_start = int(mentions[i]["start"])
                span_end = int(mentions[i]["end"])
                in_parens = (
                    0 < span_start < len(text)
                    and 0 <= span_end < len(text)
                    and text[span_start - 1] in "(["
                    and text[span_end] in ")]"
                )
                if in_parens:
                    drop_indices.add(i)

        if drop_indices:
            mentions = [m for idx, m in enumerate(mentions) if idx not in drop_indices]
            line_bounds = [b for idx, b in enumerate(line_bounds) if idx not in drop_indices]

        for idx, mention in enumerate(mentions):
            line_start, line_end = line_bounds[idx]
            line_raw = text[line_start:line_end]
            line_stripped = line_raw.strip()
            if not line_stripped:
                mention["mention_text"] = clean_mention_text_for_display(str(mention["text"]))
                mention["mention_start"] = int(mention["start"])
                mention["mention_end"] = int(mention["end"])
                continue

            if mention.get("preserve_span_text"):
                mention["mention_text"] = clean_mention_text_for_display(
                    str(mention["text"]).strip()
                )
                mention["mention_start"] = int(mention["start"])
                mention["mention_end"] = int(mention["end"])
                continue

            rel_start = max(0, int(mention["start"]) - line_start)
            rel_end = max(rel_start, int(mention["end"]) - line_start)
            clause_text = clause_around_span(line_raw, rel_start, rel_end).strip()
            if clause_text:
                clause_pos = line_raw.find(clause_text)
                if clause_pos >= 0:
                    mention["mention_text"] = clean_mention_text_for_display(clause_text)
                    mention["mention_start"] = line_start + clause_pos
                    mention["mention_end"] = line_start + clause_pos + len(clause_text)
                    continue

            left_trim = len(line_raw) - len(line_raw.lstrip())
            mention["mention_text"] = clean_mention_text_for_display(line_stripped)
            mention["mention_start"] = line_start + left_trim
            mention["mention_end"] = line_start + left_trim + len(line_stripped)
        return mentions

    chunks = [chunk.strip() for chunk in re.split(r"[;\n]+", text) if chunk.strip()]
    if len(chunks) > 1:
        fallback_mentions: List[Dict[str, object]] = []
        offset = 0
        for chunk in chunks:
            pos = text.find(chunk, offset)
            if pos < 0:
                pos = offset
            fallback_mentions.append(
                {
                    "text": chunk,
                    "norm_text": normalize_text(chunk),
                    "start": pos,
                    "end": pos + len(chunk),
                    "exact_rxcuids": [],
                    "exact_ttys": [],
                    "mention_text": clean_mention_text_for_display(chunk),
                    "mention_start": pos,
                    "mention_end": pos + len(chunk),
                }
            )
            offset = pos + len(chunk)
        return fallback_mentions

    stripped = text.strip()
    return [
        {
            "text": stripped,
            "norm_text": normalize_text(stripped),
            "start": 0,
            "end": len(stripped),
            "exact_rxcuids": [],
            "exact_ttys": [],
            "mention_text": clean_mention_text_for_display(stripped),
            "mention_start": 0,
            "mention_end": len(stripped),
        }
    ]


def top_embedding_candidates(
    mention_norm: str,
    embeddings: np.ndarray,
    rxcui_order: Sequence[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    if not mention_norm:
        return []

    query_vec = hash_embedding(mention_norm, dims=int(embeddings.shape[1]))
    similarities = np.empty(int(embeddings.shape[0]), dtype=np.float32)
    chunk_size = 50000
    for start in range(0, int(embeddings.shape[0]), chunk_size):
        end = min(start + chunk_size, int(embeddings.shape[0]))
        block = np.asarray(embeddings[start:end], dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            similarities[start:end] = block @ query_vec
    similarities = np.nan_to_num(similarities, nan=-1.0, posinf=1.0, neginf=-1.0)
    if similarities.size == 0:
        return []

    k = min(top_k, int(similarities.size))
    top_idx = np.argpartition(similarities, -k)[-k:]
    sorted_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]
    return [(rxcui_order[int(i)], float(similarities[int(i)])) for i in sorted_idx]


def choose_primary_tty(tty_map: Dict[str, str]) -> Optional[str]:
    for tty in TARGET_TTYS:
        if tty in tty_map:
            return tty
    if tty_map:
        return sorted(tty_map.keys())[0]
    return None


def chunked(values: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def fetch_neighbors_with_rela(
    conn: sqlite3.Connection, nodes: Set[str], allowed_relas: Set[str]
) -> List[Tuple[str, str]]:
    if not nodes:
        return []

    node_list = list(nodes)
    rel_list = sorted(allowed_relas)
    rows: List[Tuple[str, str]] = []
    for chunk in chunked(node_list, 800):
        src_placeholders = ",".join(["?"] * len(chunk))
        rel_placeholders = ",".join(["?"] * len(rel_list))
        sql = (
            f"SELECT DISTINCT dst, rela FROM edges WHERE src IN ({src_placeholders}) "
            f"AND rela IN ({rel_placeholders})"
        )
        params: List[str] = list(chunk) + rel_list
        for dst, rela in conn.execute(sql, tuple(params)):
            rows.append((dst, rela or ""))
    return rows


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return overlap / union if union else 0.0


def canonical_unit(unit: str) -> str:
    lowered = unit.lower()
    if lowered.startswith("unit"):
        return "unit"
    return lowered


def canonical_number(value: str) -> str:
    try:
        parsed = float(value)
    except ValueError:
        return value
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.6g}"


def add_strength_signature(signatures: Set[str], value: str, unit: str) -> None:
    u = canonical_unit(unit)
    try:
        numeric = float(value)
    except ValueError:
        signatures.add(f"{canonical_number(value)}{u}")
        return

    signatures.add(f"{canonical_number(str(numeric))}{u}")
    if u == "mg":
        mcg_value = numeric * 1000.0
        signatures.add(f"{canonical_number(str(mcg_value))}mcg")
    elif u == "mcg":
        mg_value = numeric / 1000.0
        signatures.add(f"{canonical_number(str(mg_value))}mg")


def unit_to_mg(value: float, unit: str) -> Optional[float]:
    lowered = unit.lower()
    if lowered == "mg":
        return value
    if lowered == "mcg":
        return value / 1000.0
    if lowered == "g":
        return value * 1000.0
    return None


def extract_candidate_strengths_mg(candidate_name: str) -> List[float]:
    values: List[float] = []
    for raw_value, raw_unit in SINGLE_STRENGTH_RE.findall(candidate_name.lower()):
        try:
            numeric = float(raw_value)
        except ValueError:
            continue
        mg_value = unit_to_mg(numeric, raw_unit)
        if mg_value is None:
            continue
        values.append(mg_value)
    deduped = sorted({round(v, 6) for v in values})
    return deduped


def ratio_pair_candidates_mg(first: float, second: float) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = [(first, second)]
    if first >= 10.0 and second >= 10.0:
        pairs.append((first / 1000.0, second / 1000.0))
    return pairs


def contains_close_value(values: Sequence[float], expected: float) -> bool:
    tolerance = max(0.005, abs(expected) * 0.06)
    return any(abs(v - expected) <= tolerance for v in values)


def strength_overlap_count(reference_values: Sequence[float], candidate_values: Sequence[float]) -> int:
    if not reference_values or not candidate_values:
        return 0
    return sum(1 for expected in reference_values if contains_close_value(candidate_values, expected))


def find_strength_aligned_projection(
    seed_rxcui: str,
    target_tty: str,
    required_strengths_mg: Sequence[float],
    anchor_ingredients: Set[str],
    conn: sqlite3.Connection,
    concept_ttys: Dict[str, Dict[str, str]],
    max_depth: int,
    candidate_cache: Dict[Tuple[str, str, int], List[Tuple[str, int]]],
    ingredient_cache: Dict[str, Set[str]],
) -> Optional[Dict[str, object]]:
    if not required_strengths_mg:
        return None

    candidates = collect_tty_candidates(
        start_rxcui=seed_rxcui,
        target_tty=target_tty,
        conn=conn,
        concept_ttys=concept_ttys,
        max_depth=max_depth,
        candidate_cache=candidate_cache,
    )
    best: Optional[Tuple[Tuple[int, int, int, int], str, int, str]] = None
    for candidate_rxcui, depth in candidates:
        candidate_name = concept_ttys.get(candidate_rxcui, {}).get(target_tty, "")
        if not candidate_name:
            continue

        candidate_strengths = extract_candidate_strengths_mg(candidate_name)
        overlap = strength_overlap_count(required_strengths_mg, candidate_strengths)
        if overlap == 0:
            continue

        candidate_ingredients = ingredient_set_for_rxcui(
            candidate_rxcui, conn, concept_ttys, ingredient_cache
        )
        ingredient_penalty = 0
        if anchor_ingredients:
            if candidate_ingredients == anchor_ingredients:
                ingredient_penalty = 0
            elif candidate_ingredients and candidate_ingredients.issuperset(anchor_ingredients):
                ingredient_penalty = 1
            elif candidate_ingredients and (candidate_ingredients & anchor_ingredients):
                ingredient_penalty = 2
            else:
                ingredient_penalty = 3

        rank = (-overlap, ingredient_penalty, depth, len(candidate_name))
        if best is None or rank < best[0]:
            best = (rank, candidate_rxcui, depth, candidate_name)

    if best is None:
        return None
    _rank, best_rxcui, best_depth, best_name = best
    return {
        "rxcui": best_rxcui,
        "name": best_name,
        "depth": best_depth,
    }


def in_list_from_rxcui_set(
    in_rxcuis: Set[str], concept_ttys: Dict[str, Dict[str, str]]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for in_rxcui in sorted(in_rxcuis):
        in_name = concept_ttys.get(in_rxcui, {}).get("IN")
        if not in_name:
            continue
        rows.append({"rxcui": in_rxcui, "name": in_name, "depth": 0})
    rows.sort(key=lambda row: (str(row["name"]).lower(), str(row["rxcui"])))
    return rows


def collapse_spelled_letters(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            j = i
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                j += 1
            if j - i >= 3:
                merged.append("".join(tokens[i:j]))
            else:
                merged.extend(tokens[i:j])
            i = j
            continue
        merged.append(tokens[i])
        i += 1
    return " ".join(merged)


def normalize_noisy_text(value: str) -> str:
    base = normalize_text(value)
    if not base:
        return base
    base = collapse_spelled_letters(base)
    base = re.sub(r"(?<=[a-z])(?=\d)", " ", base)
    base = re.sub(r"(?<=\d)(?=[a-z])", " ", base)
    base = re.sub(r"\b(\d+)\s*m\s*g\b", r"\1 mg", base)
    base = re.sub(r"\b(\d+)\s*m\s*c\s*g\b", r"\1 mcg", base)
    base = re.sub(r"\b(\d+)\s*m\s*l\b", r"\1 ml", base)

    tokens = base.split()
    repaired: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[0-9o]+(?:mg|mcg|g|ml|units?|meq|%)?", tok):
            tok = tok.replace("o", "0")
        replacement = NOISY_TOKEN_REPLACEMENTS.get(tok)
        if replacement:
            repaired.extend(replacement.split())
            i += 1
            continue

        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if (
                tok.isalpha()
                and nxt.isalpha()
                and tok not in CONTEXT_STOPWORDS
                and nxt not in CONTEXT_STOPWORDS
                and 2 <= len(tok) <= 6
                and 2 <= len(nxt) <= 8
            ):
                joined = tok + nxt
                if joined in NOISY_TOKEN_REPLACEMENTS:
                    repaired.extend(NOISY_TOKEN_REPLACEMENTS[joined].split())
                    i += 2
                    continue
                if len(joined) >= 7:
                    repaired.append(joined)
                    i += 2
                    continue

        repaired.append(tok)
        i += 1

    return " ".join(repaired)


def strip_context_tokens(text_norm: str) -> str:
    tokens = [tok for tok in text_norm.split() if tok not in CONTEXT_STOPWORDS]
    return " ".join(tokens)


def parse_pipe_med_fields(text: str) -> Optional[Dict[str, str]]:
    stripped = text.strip()
    if not stripped.lower().startswith("med|"):
        return None
    parts = [part.strip() for part in stripped.split("|")]
    if len(parts) < 6:
        return None
    return {
        "name_raw": parts[1],
        "strength_raw": parts[2],
        "route_norm": normalize_text(parts[3]),
        "schedule_norm": normalize_text(parts[4]),
        "status_norm": normalize_text(parts[5]),
    }


def parse_rxe_fields(text: str) -> Optional[Dict[str, str]]:
    stripped = text.strip()
    if not stripped.lower().startswith("rxe|"):
        return None
    parts = [part.strip() for part in stripped.split("|")]
    if len(parts) < 3:
        return None
    payload = parts[-1]
    payload_parts = [part.strip() for part in payload.split("^")]
    if len(payload_parts) < 4:
        return None
    return {
        "name_raw": payload_parts[0],
        "strength_raw": payload_parts[1],
        "route_norm": normalize_text(payload_parts[2]),
        "schedule_norm": normalize_text(payload_parts[3]),
        "status_norm": "",
    }


def has_explicit_combo_hint(raw_text: str, mention_norm: str) -> bool:
    if "+" in raw_text:
        return True
    padded_norm = f" {mention_norm} "
    if " and " in padded_norm or " with " in padded_norm:
        return True

    lowered = raw_text.lower()
    if re.search(r"\b[a-z]{3,}\s*/\s*[a-z]{3,}\b", lowered):
        return True
    if re.search(r"\b[a-z]{3,}\s*-\s*[a-z]{3,}\b", lowered):
        return True
    return False


def lookup_exact_alias_rows(
    conn: sqlite3.Connection, lookup_norm: str, max_exact_candidates: int
) -> List[Tuple[str, str]]:
    if not lookup_norm or lookup_norm in NON_DRUG_EXACT_TERMS:
        return []
    rows = conn.execute(
        "SELECT DISTINCT rxcui, tty FROM alias WHERE norm_text = ? LIMIT ?",
        (lookup_norm, max_exact_candidates),
    ).fetchall()
    if not rows:
        return []
    if all((tty or "").upper() in NON_DRUG_EXACT_TTYS for _, tty in rows):
        return []
    return rows


def iter_line_spans(text: str) -> Iterable[Tuple[int, int, str]]:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_text = raw_line.rstrip("\n")
        start = offset
        end = start + len(line_text)
        yield start, end, line_text
        offset += len(raw_line)
    if text and not text.endswith("\n"):
        return
    if not text:
        return
    if text.endswith("\n"):
        yield len(text), len(text), ""


def should_scan_dense_line(line_text: str) -> bool:
    stripped = line_text.strip()
    if len(stripped) < 70:
        return False
    canonical = normalize_noisy_text(stripped)
    if len(canonical) < 50:
        return False
    match_count = len(list(DENSE_LINE_MED_RE.finditer(canonical)))
    if match_count < 2:
        return False

    raw_tokens = stripped.split()
    if len(raw_tokens) <= 4:
        return True

    mixed_alnum_tokens = sum(
        1
        for tok in raw_tokens
        if re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok)
    )
    long_tokens = sum(1 for tok in raw_tokens if len(tok) >= 12)
    return mixed_alnum_tokens >= 2 or long_tokens >= 3


def extract_dense_line_mentions(
    line_text: str,
    line_start: int,
    conn: sqlite3.Connection,
    max_exact_candidates: int,
) -> List[Dict[str, object]]:
    raw_canonical_line = "".join(CONFUSABLE_CHAR_MAP.get(ch, ch) for ch in line_text).lower()
    matches = list(DENSE_LINE_MED_RE.finditer(raw_canonical_line))
    if not matches:
        return []
    mentions: List[Dict[str, object]] = []
    for idx, match in enumerate(matches):
        seg_start = int(match.start())
        seg_end = int(matches[idx + 1].start()) if idx + 1 < len(matches) else len(line_text)
        segment = line_text[seg_start:seg_end].strip(" ,;")
        if len(segment) < 3:
            continue

        normalized_segment = normalize_noisy_text(segment)
        core_raw = DENSE_LINE_MED_TOKENS.get(match.group(0).lower(), match.group(0).lower())
        core_norm = normalize_noisy_text(core_raw)
        rows = lookup_exact_alias_rows(conn, core_norm, max_exact_candidates)
        if not rows:
            rows = lookup_exact_alias_rows(
                conn, strip_context_tokens(normalized_segment), max_exact_candidates
            )

        mentions.append(
            {
                "text": segment,
                "norm_text": normalized_segment or core_norm,
                "start": line_start + seg_start,
                "end": line_start + seg_start + len(segment),
                "exact_rxcuids": [row[0] for row in rows],
                "exact_ttys": sorted({row[1] for row in rows if row[1]}),
                "preserve_span_text": True,
            }
        )
    return mentions


def allow_multi_token_span(span_text: str) -> bool:
    lowered = span_text.lower()
    if "," in span_text and not any(
        marker in lowered for marker in (" and ", " with ", "/", "+")
    ):
        return False
    return True


def build_query_variants(raw_text: str, mention_text: str, mention_norm: str) -> List[str]:
    variants: List[str] = []

    def add_variant(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    add_variant(mention_norm)
    add_variant(normalize_noisy_text(raw_text))
    add_variant(normalize_noisy_text(mention_text))
    for item in list(variants):
        add_variant(strip_context_tokens(item))
    return variants


def strength_signatures(text_norm: str) -> Set[str]:
    working = text_norm.lower()
    working = re.sub(r"[^a-z0-9./%+\-\s]+", " ", working)
    working = re.sub(r"\s+", " ", working).strip()
    signatures: Set[str] = set()
    for first, second, unit in RATIO_STRENGTH_RE.findall(working):
        add_strength_signature(signatures, first, unit)
        add_strength_signature(signatures, second, unit)
    for value, unit in SINGLE_STRENGTH_RE.findall(working):
        add_strength_signature(signatures, value, unit)
    return signatures


def has_surface_strength_signal(text_norm: str) -> bool:
    if not text_norm:
        return False
    if STRENGTH_RE.search(text_norm):
        return True
    if RATIO_STRENGTH_RE.search(text_norm):
        return True
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:u|unit|units)\s*(?:/|\s+)\s*kg\s*(?:/|\s+)\s*h(?:r)?\b",
        text_norm,
    ):
        return True
    return False


def has_implicit_numeric_dose_signal(
    mention_surface_norm: str, mention_features: Dict[str, object]
) -> bool:
    if not mention_surface_norm:
        return False
    if has_surface_strength_signal(mention_surface_norm):
        return True

    if bool(mention_features.get("iv_route_hint", False)):
        return False
    if bool(mention_features.get("continuous_infusion_hint", False)):
        return False
    if re.search(r"\b(?:kg|hr|h|min|sec)\b", mention_surface_norm):
        return False

    numeric_values: List[float] = []
    for raw in BARE_NUMBER_RE.findall(mention_surface_norm):
        try:
            numeric = float(raw)
        except ValueError:
            continue
        if numeric <= 0.0 or numeric > 1500.0:
            continue
        numeric_values.append(numeric)
    if not numeric_values:
        return False

    form_tokens = mention_features.get("form_tokens", set())
    has_oral_form_hint = (
        isinstance(form_tokens, set)
        and any(tok in form_tokens for tok in {"tablet", "capsule", "er", "ec", "xl24", "oral"})
    )
    schedule_hint = bool(mention_features.get("schedule_hint", False))
    oral_route_hint = bool(mention_features.get("oral_route_hint", False))

    return schedule_hint or oral_route_hint or has_oral_form_hint


def candidate_form_flags(candidate_norm: str) -> Set[str]:
    flags: Set[str] = set()
    padded = f" {candidate_norm} "
    for form_key, hints in FORM_HINTS.items():
        for hint in hints:
            if f" {hint} " in padded:
                flags.add(form_key)
                break
    if " oral " in padded:
        flags.add("oral")
    return flags


def find_segment_bounds(text: str, start: int, end: int, pattern: str) -> Tuple[int, int]:
    left = 0
    right = len(text)
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if match.end() <= start:
            left = match.end()
            continue
        if match.start() >= end:
            right = match.start()
            break
    return left, right


def focused_context_around_match(full_text: str, match_start: int, match_end: int) -> str:
    line_start, line_end = get_line_bounds(full_text, match_start, match_end)
    line_text = full_text[line_start:line_end]
    rel_start = max(0, match_start - line_start)
    rel_end = max(rel_start, match_end - line_start)

    seg_start = 0
    seg_end = len(line_text)
    for pattern in SEGMENT_SPLIT_PATTERNS:
        local = line_text[seg_start:seg_end]
        local_start = rel_start - seg_start
        local_end = rel_end - seg_start
        left, right = find_segment_bounds(local, local_start, local_end, pattern)
        seg_start += left
        seg_end = seg_start + (right - left)

    focused = line_text[seg_start:seg_end].strip()
    if len(focused) < 3:
        return normalize_noisy_text(line_text.strip())
    return normalize_noisy_text(focused)


def extract_mention_features(
    mention_text: str, mention_norm: str, context_norm: str
) -> Dict[str, object]:
    strength_tokens = {m.group(0).strip() for m in STRENGTH_RE.finditer(context_norm)}
    strength_sigs = strength_signatures(context_norm)
    pipe_fields = parse_pipe_med_fields(mention_text)
    rxe_fields = parse_rxe_fields(mention_text)
    structured_fields = pipe_fields or rxe_fields
    structured_med_record = structured_fields is not None
    form_tokens: Set[str] = set()
    padded_norm = f" {context_norm} "
    for form_key, hints in FORM_HINTS.items():
        for hint in hints:
            if f" {hint} " in padded_norm:
                form_tokens.add(form_key)
                break

    combo_probe_text = structured_fields["name_raw"] if structured_fields else mention_text
    combo_hint = has_explicit_combo_hint(combo_probe_text, mention_norm)

    iv_route_hint = bool(re.search(r"\b(?:iv|ivp|ivpb|intravenous)\b", context_norm))
    oral_route_hint = bool(re.search(r"\b(?:po|oral)\b", context_norm))
    continuous_infusion_hint = bool(
        re.search(r"\b(?:gtt|drip|infusion|cont|continuous)\b", context_norm)
    )
    infusion_rate_hint = bool(
        re.search(
            r"\b(?:mcg|mg|u|unit|units)\s*(?:/|\s+)?\s*(?:kg\s*(?:/|\s+)?\s*h(?:r)?|kgh?r)\w*",
            context_norm,
        )
    )
    if structured_fields:
        route_norm = structured_fields["route_norm"]
        schedule_norm = structured_fields["schedule_norm"]
        if route_norm in IV_ROUTE_TOKENS:
            iv_route_hint = True
            form_tokens.add("injection")
        if route_norm in ORAL_ROUTE_TOKENS:
            oral_route_hint = True
        if schedule_norm in {"cont", "continuous"}:
            continuous_infusion_hint = True
    if infusion_rate_hint:
        continuous_infusion_hint = True
        iv_route_hint = True
        form_tokens.add("injection")

    ratio_pairs: List[Tuple[float, float]] = []
    ascii_mention = (
        unicodedata.normalize("NFKD", mention_text).encode("ascii", "ignore").decode("ascii")
    )
    for first_raw, second_raw in SLASH_RATIO_RE.findall(ascii_mention.lower()):
        try:
            first_val = float(first_raw)
            second_val = float(second_raw)
        except ValueError:
            continue
        if first_val <= 0.0 or second_val <= 0.0:
            continue
        ratio_pairs.append((first_val, second_val))

    numeric_strength_values: List[float] = []
    for raw in BARE_NUMBER_RE.findall(context_norm):
        try:
            numeric = float(raw)
        except ValueError:
            continue
        if numeric <= 0.0 or numeric > 1500.0:
            continue
        numeric_strength_values.append(numeric)
    numeric_strength_values = sorted({round(v, 6) for v in numeric_strength_values})

    schedule_hint = bool(
        re.search(
            r"\b(?:daily|qd|qam|qhs|bid|tid|qid|nightly|weekly|monthly|q\d+h)\b",
            context_norm,
        )
    )

    return {
        "strength_tokens": strength_tokens,
        "strength_sigs": strength_sigs,
        "form_tokens": form_tokens,
        "combo_hint": combo_hint,
        "structured_med_record": structured_med_record,
        "iv_route_hint": iv_route_hint,
        "oral_route_hint": oral_route_hint,
        "continuous_infusion_hint": continuous_infusion_hint,
        "ratio_pairs": ratio_pairs,
        "numeric_strength_values": numeric_strength_values,
        "schedule_hint": schedule_hint,
    }


def get_line_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start < 0 else line_start + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    return line_start, line_end


def clause_around_span(line_text: str, rel_start: int, rel_end: int) -> str:
    left_candidates = [line_text.rfind(ch, 0, rel_start) for ch in ".;:!?,"]
    left_bound = max(left_candidates) if left_candidates else -1
    right_candidates = [line_text.find(ch, rel_end) for ch in ".;:!?,"]
    right_candidates = [pos for pos in right_candidates if pos >= 0]
    right_bound = min(right_candidates) if right_candidates else len(line_text)
    return line_text[left_bound + 1 : right_bound].strip()


def clean_mention_text_for_display(text: str) -> str:
    stripped = text.strip()
    if "<" not in stripped or "value" not in stripped.lower():
        return stripped

    values: List[str] = []
    for _quote, captured in XML_VALUE_ATTR_RE.findall(stripped):
        value = html.unescape(captured).strip()
        if value:
            values.append(value)
    if not values:
        return stripped

    generic_values = {"active", "order", "true", "false"}
    meaningful: List[str] = []
    for value in values:
        lowered = value.lower()
        if lowered in generic_values:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            continue
        if re.fullmatch(r"[a-z]+/\d+", lowered):
            continue
        if not any(ch.isalpha() for ch in value):
            continue
        meaningful.append(value)

    if meaningful:
        return max(meaningful, key=len)
    return max(values, key=len)


def is_negated_mention(
    full_text: str, span_start: int, span_end: int, mention_norm: str
) -> bool:
    line_start, line_end = get_line_bounds(full_text, span_start, span_end)
    line_text = full_text[line_start:line_end]
    if re.match(r"^\s*med\|", line_text, flags=re.IGNORECASE):
        return False
    rel_start = max(0, span_start - line_start)
    rel_end = max(rel_start, span_end - line_start)
    clause = clause_around_span(line_text, rel_start, rel_end)
    clause_norm = normalize_text(clause)
    if not clause_norm:
        return False

    tokens = mention_norm.split()
    if not tokens:
        return False
    anchor = re.escape(tokens[0])
    neg_pattern = (
        rf"\b(?:no|denies|deny|denied|without|not taking|not on|off|hold|held|stop|"
        rf"stopped|discontinue|discontinued|allergic to|allergy to)\b"
        rf"(?:\s+\w+){{0,8}}\s+\b{anchor}\b"
    )
    neg_match = re.search(neg_pattern, clause_norm)
    if neg_match:
        neg_span = clause_norm[neg_match.start() : neg_match.end()]
        if not re.search(r"\b(?:but|however|though|although|except)\b", neg_span):
            return True

    post_neg_pattern = rf"\b{anchor}\b(?:\s+\w+){{0,6}}\s+\b(?:held|stopped|discontinued)\b"
    return bool(re.search(post_neg_pattern, clause_norm))


def has_brand_evidence(
    conn: sqlite3.Connection,
    mention_text: str,
    exact_ttys: Sequence[str],
    best_tty: Optional[str],
) -> bool:
    if best_tty in {"BN", "SBD", "BPCK"}:
        return True
    if any(tty in {"BN", "SBD", "BPCK"} for tty in exact_ttys):
        return True

    chunks = re.findall(r"[\(\[]([^)\]]{2,60})[\)\]]", mention_text)
    for chunk in chunks:
        norm_chunk = normalize_text(chunk)
        if not norm_chunk:
            continue
        row = conn.execute(
            "SELECT 1 FROM alias WHERE norm_text = ? AND tty = 'BN' LIMIT 1",
            (norm_chunk,),
        ).fetchone()
        if row:
            return True
    return False


def collect_tty_candidates(
    start_rxcui: str,
    target_tty: str,
    conn: sqlite3.Connection,
    concept_ttys: Dict[str, Dict[str, str]],
    max_depth: int,
    candidate_cache: Dict[Tuple[str, str, int], List[Tuple[str, int]]],
) -> List[Tuple[str, int]]:
    cache_key = (start_rxcui, target_tty, max_depth)
    cached = candidate_cache.get(cache_key)
    if cached is not None:
        return cached

    allowed_relas = TARGET_RELAS.get(target_tty, set())
    visited_depth: Dict[str, int] = {start_rxcui: 0}
    frontier: Set[str] = {start_rxcui}
    depth = 0

    while frontier and depth < max_depth:
        edge_rows = fetch_neighbors_with_rela(conn, frontier, allowed_relas)
        next_frontier: Set[str] = set()
        for dst, _rela in edge_rows:
            if dst in visited_depth:
                continue
            visited_depth[dst] = depth + 1
            next_frontier.add(dst)
        frontier = next_frontier
        depth += 1

    candidates: List[Tuple[str, int]] = []
    for rxcui, node_depth in visited_depth.items():
        if target_tty in concept_ttys.get(rxcui, {}):
            candidates.append((rxcui, node_depth))

    candidate_cache[cache_key] = candidates
    return candidates


def ingredient_set_for_rxcui(
    rxcui: str,
    conn: sqlite3.Connection,
    concept_ttys: Dict[str, Dict[str, str]],
    ingredient_cache: Dict[str, Set[str]],
) -> Set[str]:
    cached = ingredient_cache.get(rxcui)
    if cached is not None:
        return cached

    start_ttys = concept_ttys.get(rxcui, {})
    if "IN" in start_ttys:
        ingredient_cache[rxcui] = {rxcui}
        return {rxcui}

    def direct_in_neighbors(nodes: Set[str]) -> Set[str]:
        if not nodes:
            return set()
        rel_rows = fetch_neighbors_with_rela(conn, nodes, DIRECT_INGREDIENT_RELAS)
        return {dst for dst, _rela in rel_rows if "IN" in concept_ttys.get(dst, {})}

    direct_hits = direct_in_neighbors({rxcui})
    if direct_hits:
        ingredient_cache[rxcui] = direct_hits
        return direct_hits

    bridge_first_hop = {
        dst
        for dst, _rela in fetch_neighbors_with_rela(conn, {rxcui}, INGREDIENT_BRIDGE_RELAS)
    }
    found_in = direct_in_neighbors(bridge_first_hop)
    if found_in:
        ingredient_cache[rxcui] = found_in
        return found_in

    second_hop_seed = {
        node
        for node in bridge_first_hop
        if any(
            tty in concept_ttys.get(node, {})
            for tty in {"SBD", "SCD", "BN", "GPCK", "BPCK"}
        )
    }
    bridge_second_hop = {
        dst
        for dst, _rela in fetch_neighbors_with_rela(
            conn,
            second_hop_seed,
            {"consists_of", "constitutes", "has_part", "part_of", "contains", "contained_in"},
        )
    }
    found_in |= direct_in_neighbors(bridge_second_hop)

    ingredient_cache[rxcui] = found_in
    return found_in


def score_tty_candidate(
    mention_norm: str,
    mention_features: Dict[str, object],
    target_tty: str,
    candidate_name: str,
    depth: int,
    anchor_ingredients: Set[str],
    candidate_ingredients: Set[str],
    preferred_strengths_mg: Optional[Sequence[float]] = None,
) -> float:
    candidate_norm = normalize_text(candidate_name)
    candidate_flags = candidate_form_flags(candidate_norm)
    candidate_strengths_mg = extract_candidate_strengths_mg(candidate_name)
    lexical = token_jaccard(mention_norm, candidate_norm)
    score = 1.8 * lexical
    score -= 0.45 * float(depth)

    strength_tokens = mention_features["strength_tokens"]  # type: ignore[assignment]
    strength_sigs = mention_features["strength_sigs"]  # type: ignore[assignment]
    ratio_pairs = mention_features.get("ratio_pairs", [])
    numeric_strength_values = mention_features.get("numeric_strength_values", [])
    schedule_hint = bool(mention_features.get("schedule_hint", False))
    form_tokens = mention_features["form_tokens"]  # type: ignore[assignment]
    combo_hint = bool(mention_features["combo_hint"])
    structured_med_record = bool(mention_features.get("structured_med_record", False))
    iv_route_hint = bool(mention_features.get("iv_route_hint", False))
    oral_route_hint = bool(mention_features.get("oral_route_hint", False))
    continuous_infusion_hint = bool(
        mention_features.get("continuous_infusion_hint", False)
    )
    route_sensitive_tty = target_tty in {"SBD", "SCD", "GPCK", "BPCK", "BN", "SCDC"}

    if isinstance(strength_tokens, set) and strength_tokens:
        matched = sum(1 for token in strength_tokens if token in candidate_norm)
        if matched > 0:
            score += 1.8 * float(matched)
        else:
            score -= 0.6

    if isinstance(strength_sigs, set) and strength_sigs:
        candidate_sigs = strength_signatures(candidate_name)
        if candidate_sigs:
            overlap = len(strength_sigs & candidate_sigs)
            if overlap > 0:
                score += 2.2 * float(overlap)
            else:
                score -= 1.4
        else:
            score -= 0.8

    if isinstance(ratio_pairs, list) and ratio_pairs:
        matched_ratio = False
        if len(candidate_strengths_mg) >= 2:
            for first_val, second_val in ratio_pairs:
                pair_matched = False
                for cand_first, cand_second in ratio_pair_candidates_mg(first_val, second_val):
                    first_ok = contains_close_value(candidate_strengths_mg, cand_first)
                    second_ok = contains_close_value(candidate_strengths_mg, cand_second)
                    if first_ok and second_ok:
                        pair_matched = True
                        break
                if pair_matched:
                    matched_ratio = True
                    break

        if matched_ratio:
            score += 3.4
        elif len(candidate_strengths_mg) >= 2:
            score -= 1.3

    if isinstance(numeric_strength_values, list) and numeric_strength_values and candidate_strengths_mg:
        numeric_matches = 0
        for expected in numeric_strength_values:
            if contains_close_value(candidate_strengths_mg, expected):
                numeric_matches += 1
        if numeric_matches > 0:
            score += 1.25 * float(min(numeric_matches, 2))
        elif not (
            isinstance(ratio_pairs, list) and ratio_pairs
        ):
            score -= 0.4

    if preferred_strengths_mg:
        if candidate_strengths_mg:
            preferred_overlap = strength_overlap_count(
                preferred_strengths_mg, candidate_strengths_mg
            )
            if preferred_overlap > 0:
                score += 2.0 * float(min(preferred_overlap, 2))
            else:
                score -= 2.2
        else:
            score -= 0.5

    if isinstance(form_tokens, set) and form_tokens:
        matched_form = 0
        for form_key in form_tokens:
            if form_key in candidate_flags:
                matched_form += 1
        if matched_form > 0:
            score += 1.2 * float(matched_form)
        else:
            score -= 0.9

        if "inhaler" in form_tokens and (
            "tablet" in candidate_flags
            or "capsule" in candidate_flags
            or "oral" in candidate_flags
            or "injection" in candidate_flags
        ):
            score -= 2.4
        if "inhaler" in form_tokens and "inhaler" in candidate_flags:
            score += 2.4
        if "inhaler" in form_tokens and "inhaler" not in candidate_flags:
            score -= 3.2
        if "inhaler" in form_tokens and "solution" in candidate_flags and "inhaler" not in candidate_flags:
            score -= 2.2
        if "dry_powder" in form_tokens and "dry_powder" in candidate_flags:
            score += 0.8

        if "injection" in form_tokens and (
            "tablet" in candidate_flags
            or "capsule" in candidate_flags
            or "oral" in candidate_flags
            or "inhaler" in candidate_flags
        ):
            score -= 2.4

        if ("tablet" in form_tokens or "capsule" in form_tokens) and "inhaler" in candidate_flags:
            score -= 2.0

    if schedule_hint and "injection" in candidate_flags and not (
        isinstance(form_tokens, set) and "injection" in form_tokens
    ):
        score -= 2.4
    if schedule_hint and any(
        form in candidate_flags for form in {"oral", "tablet", "capsule", "dr", "er"}
    ):
        score += 0.6

    if iv_route_hint and route_sensitive_tty:
        if "injection" in candidate_flags:
            score += 3.2
        else:
            score -= 4.6
        if "patch" in candidate_flags or "sublingual" in candidate_flags:
            score -= 3.5

    if oral_route_hint and route_sensitive_tty:
        if any(
            form in candidate_flags for form in {"oral", "tablet", "capsule", "dr", "er"}
        ):
            score += 0.9
        if "injection" in candidate_flags:
            score -= 3.2
        if "tablet" in candidate_flags:
            score += 0.9
        if "capsule" in candidate_flags and not (
            isinstance(form_tokens, set) and "capsule" in form_tokens
        ):
            score -= 1.1

    if continuous_infusion_hint and iv_route_hint and route_sensitive_tty:
        if "injection" in candidate_flags:
            score += 0.9
        if "patch" in candidate_flags or "sublingual" in candidate_flags:
            score -= 2.8

    if "dry powder" in candidate_norm and not (
        isinstance(form_tokens, set) and "dry_powder" in form_tokens
    ):
        score -= 1.8

    if "solution" in candidate_flags and not (
        isinstance(form_tokens, set)
        and any(form in form_tokens for form in {"solution", "inhaler", "injection"})
    ):
        score -= 0.9
    if " granules " in f" {candidate_norm} " and not (
        isinstance(form_tokens, set) and "suspension" in form_tokens
    ):
        score -= 0.8

    combo_allowed = combo_hint or len(anchor_ingredients) > 1
    if target_tty == "MIN" and structured_med_record and not combo_hint:
        return -1000.0
    if target_tty == "MIN" and not combo_allowed:
        return -1000.0

    if anchor_ingredients:
        if candidate_ingredients:
            if candidate_ingredients == anchor_ingredients:
                score += 2.4
            elif candidate_ingredients.issuperset(anchor_ingredients):
                score += 0.3
                if not combo_allowed:
                    score -= 4.0
            elif candidate_ingredients & anchor_ingredients:
                score += 0.1
                if not combo_allowed and len(candidate_ingredients) > len(anchor_ingredients):
                    score -= 3.2
            else:
                score -= 3.0
        else:
            score -= 0.4

        if candidate_ingredients and len(candidate_ingredients) > len(anchor_ingredients):
            extra_count = len(candidate_ingredients) - len(anchor_ingredients)
            score -= 1.6 * float(extra_count)

        if not combo_allowed and len(candidate_ingredients) > len(anchor_ingredients):
            score -= 2.0

    if not combo_allowed and "/" in candidate_norm and target_tty in {
        "SBD",
        "SCD",
        "SCDC",
        "BN",
        "GPCK",
        "BPCK",
    }:
        score -= 3.0

    if target_tty in {"GPCK", "BPCK"} and " pack " not in f" {candidate_norm} ":
        score -= 1.0

    return score


def project_ttys(
    start_rxcui: str,
    mention_text: str,
    mention_norm: str,
    mention_context_norm: str,
    conn: sqlite3.Connection,
    concept_ttys: Dict[str, Dict[str, str]],
    max_depth: int,
    candidate_cache: Dict[Tuple[str, str, int], List[Tuple[str, int]]],
    ingredient_cache: Dict[str, Set[str]],
) -> Dict[str, object]:
    projected: Dict[str, object] = {tty: None for tty in TARGET_TTYS}
    mention_features = extract_mention_features(
        mention_text, mention_norm, mention_context_norm
    )
    mention_surface_norm = normalize_noisy_text(mention_text)
    mention_surface_has_dose_signal = has_implicit_numeric_dose_signal(
        mention_surface_norm, mention_features
    )
    structured_med_record = bool(mention_features.get("structured_med_record", False))
    start_ttys = concept_ttys.get(start_rxcui, {})
    ingredient_anchor = "IN" in start_ttys
    anchor_ingredients = ingredient_set_for_rxcui(
        start_rxcui, conn, concept_ttys, ingredient_cache
    )

    for target_tty in TARGET_TTYS:
        if (
            target_tty in {"SCD", "SBD", "SCDC"}
            and ingredient_anchor
            and not mention_surface_has_dose_signal
            and not structured_med_record
        ):
            continue
        preferred_strengths_mg: Optional[List[float]] = None
        if target_tty == "SCD" and projected["SBD"] is not None:
            preferred_strengths_mg = extract_candidate_strengths_mg(
                str(projected["SBD"]["name"])
            )
        elif target_tty == "SCDC":
            if projected["SCD"] is not None:
                preferred_strengths_mg = extract_candidate_strengths_mg(
                    str(projected["SCD"]["name"])
                )
            elif projected["SBD"] is not None:
                preferred_strengths_mg = extract_candidate_strengths_mg(
                    str(projected["SBD"]["name"])
                )

        candidates = collect_tty_candidates(
            start_rxcui=start_rxcui,
            target_tty=target_tty,
            conn=conn,
            concept_ttys=concept_ttys,
            max_depth=max_depth,
            candidate_cache=candidate_cache,
        )
        if not candidates:
            continue

        scored: List[Tuple[float, int, str, str]] = []
        for candidate_rxcui, depth in candidates:
            candidate_name = concept_ttys.get(candidate_rxcui, {}).get(target_tty, "")
            if not candidate_name:
                continue
            candidate_ingredients = ingredient_set_for_rxcui(
                candidate_rxcui, conn, concept_ttys, ingredient_cache
            )
            score = score_tty_candidate(
                mention_norm=mention_norm,
                mention_features=mention_features,
                target_tty=target_tty,
                candidate_name=candidate_name,
                depth=depth,
                anchor_ingredients=anchor_ingredients,
                candidate_ingredients=candidate_ingredients,
                preferred_strengths_mg=preferred_strengths_mg,
            )
            scored.append((score, depth, candidate_rxcui, candidate_name))

        if not scored:
            continue

        best_score, best_depth, best_rxcui, best_name = max(
            scored, key=lambda item: (item[0], -item[1])
        )
        if best_score < -2.5:
            continue

        projected[target_tty] = {
            "rxcui": best_rxcui,
            "name": best_name,
            "depth": best_depth,
        }

    if projected["IN"] is None and len(anchor_ingredients) == 1:
        inferred_in_rxcui = next(iter(anchor_ingredients))
        inferred_in_name = concept_ttys.get(inferred_in_rxcui, {}).get("IN")
        if inferred_in_name:
            projected["IN"] = {
                "rxcui": inferred_in_rxcui,
                "name": inferred_in_name,
                "depth": 0,
            }

    if projected["SCD"] is not None and projected["SCDC"] is not None:
        scd_strengths = extract_candidate_strengths_mg(str(projected["SCD"]["name"]))
        scdc_strengths = extract_candidate_strengths_mg(str(projected["SCDC"]["name"]))
        if (
            scd_strengths
            and scdc_strengths
            and strength_overlap_count(scd_strengths, scdc_strengths) == 0
        ):
            aligned_scdc = find_strength_aligned_projection(
                seed_rxcui=str(projected["SCD"]["rxcui"]),
                target_tty="SCDC",
                required_strengths_mg=scd_strengths,
                anchor_ingredients=anchor_ingredients,
                conn=conn,
                concept_ttys=concept_ttys,
                max_depth=max_depth,
                candidate_cache=candidate_cache,
                ingredient_cache=ingredient_cache,
            )
            projected["SCDC"] = aligned_scdc

    if projected["MIN"] is not None:
        min_rxcui = str(projected["MIN"]["rxcui"])
        min_ingredients = ingredient_set_for_rxcui(
            min_rxcui, conn, concept_ttys, ingredient_cache
        )
        if len(min_ingredients) < 2:
            min_name_norm = normalize_noisy_text(str(projected["MIN"]["name"]))
            part_candidates = re.split(r"\s*(?:/|\+|\band\b)\s*", min_name_norm)
            for part in part_candidates:
                part_norm = strip_context_tokens(part)
                if not part_norm:
                    continue
                part_norm = re.sub(r"\b\d+(?:\.\d+)?\b", " ", part_norm)
                part_norm = re.sub(r"\s+", " ", part_norm).strip()
                if not part_norm:
                    continue
                row = conn.execute(
                    "SELECT rxcui FROM alias WHERE norm_text = ? AND tty = 'IN' LIMIT 1",
                    (part_norm,),
                ).fetchone()
                if row:
                    min_ingredients.add(str(row[0]))
        in_all = in_list_from_rxcui_set(min_ingredients, concept_ttys)
        if in_all:
            projected["IN_ALL"] = in_all
            projected["IN"] = list(in_all)
    elif len(anchor_ingredients) > 1:
        anchor_in_all = in_list_from_rxcui_set(anchor_ingredients, concept_ttys)
        if anchor_in_all:
            projected["IN_ALL"] = anchor_in_all
            projected["IN"] = list(anchor_in_all)

    return projected


def load_concept_ttys(conn: sqlite3.Connection) -> Dict[str, Dict[str, str]]:
    concept_ttys: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in conn.execute("SELECT rxcui, tty, name FROM concept_tty"):
        concept_ttys[row[0]][row[1]] = row[2]
    return concept_ttys


def load_preferred_names(conn: sqlite3.Connection) -> Dict[str, str]:
    return {row[0]: row[1] for row in conn.execute("SELECT rxcui, preferred_name FROM concepts")}


def infer_text_with_resources(
    input_text: str,
    conn: sqlite3.Connection,
    embeddings: np.ndarray,
    rxcui_order: Sequence[str],
    concept_ttys: Dict[str, Dict[str, str]],
    preferred_names: Dict[str, str],
    top_k: int = 40,
    exact_boost: float = 0.35,
    max_graph_depth: int = 3,
    max_ngram: int = 8,
    max_exact_candidates: int = 25,
    candidate_cache: Optional[Dict[Tuple[str, str, int], List[Tuple[str, int]]]] = None,
    ingredient_cache: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, object]:
    mentions = detect_mentions(
        text=input_text,
        conn=conn,
        max_ngram=max_ngram,
        max_exact_candidates=max_exact_candidates,
    )
    results: List[Dict[str, object]] = []
    if candidate_cache is None:
        candidate_cache = {}
    if ingredient_cache is None:
        ingredient_cache = {}

    for mention in mentions:
        mention_text = str(mention.get("mention_text", mention["text"]))
        mention_norm = str(mention["norm_text"])
        matched_text = str(mention["text"])
        projection_norm = strip_context_tokens(normalize_noisy_text(matched_text))
        if not projection_norm:
            projection_norm = mention_norm
        mention_start = int(mention.get("mention_start", mention["start"]))
        mention_end = int(mention.get("mention_end", mention["end"]))
        match_start = int(mention["start"])
        match_end = int(mention["end"])
        line_start, line_end = get_line_bounds(input_text, mention_start, mention_end)
        line_context = input_text[line_start:line_end].strip()
        focused_context_norm = focused_context_around_match(
            input_text, match_start, match_end
        )
        if len(focused_context_norm) >= 3:
            mention_context_norm = focused_context_norm
        elif len(line_context) >= 4:
            mention_context_norm = normalize_noisy_text(line_context)
        else:
            context_start = max(0, mention_start - 24)
            context_end = min(len(input_text), mention_end + 40)
            mention_context_norm = normalize_noisy_text(input_text[context_start:context_end])
        exact_rxcuids: List[str] = list(mention["exact_rxcuids"])  # type: ignore[assignment]
        exact_ttys: List[str] = list(mention.get("exact_ttys", []))  # type: ignore[assignment]

        if is_negated_mention(
            full_text=input_text,
            span_start=match_start,
            span_end=match_end,
            mention_norm=mention_norm,
        ):
            continue

        score_by_rxcui: Dict[str, float] = {}
        query_variants = build_query_variants(
            raw_text=matched_text,
            mention_text=mention_text,
            mention_norm=mention_norm,
        )
        for query in query_variants:
            query_penalty = 0.0 if query == mention_norm else 0.03
            for rxcui, score in top_embedding_candidates(
                mention_norm=query,
                embeddings=embeddings,
                rxcui_order=rxcui_order,
                top_k=top_k,
            ):
                adjusted = score - query_penalty
                current = score_by_rxcui.get(rxcui)
                if current is None or adjusted > current:
                    score_by_rxcui[rxcui] = adjusted

        for rxcui in exact_rxcuids:
            score_by_rxcui[rxcui] = score_by_rxcui.get(rxcui, 0.0) + exact_boost

        if not score_by_rxcui:
            continue

        best_rxcui, best_score = max(score_by_rxcui.items(), key=lambda item: item[1])
        has_strength_signal = bool(
            STRENGTH_RE.search(mention_context_norm)
            or RATIO_STRENGTH_RE.search(mention_context_norm)
            or SLASH_RATIO_RE.search(mention_context_norm)
        )
        if not has_strength_signal and exact_rxcuids:
            for exact_rxcui in exact_rxcuids:
                if "IN" in concept_ttys.get(exact_rxcui, {}):
                    best_rxcui = exact_rxcui
                    best_score = score_by_rxcui.get(exact_rxcui, best_score)
                    break
        if not has_strength_signal:
            in_override_rxcui: Optional[str] = None
            for lookup_norm in (
                mention_norm,
                projection_norm,
                strip_context_tokens(mention_norm),
            ):
                if not lookup_norm:
                    continue
                row = conn.execute(
                    "SELECT rxcui FROM alias WHERE norm_text = ? AND tty = 'IN' LIMIT 1",
                    (lookup_norm,),
                ).fetchone()
                if row:
                    in_override_rxcui = str(row[0])
                    break
            if in_override_rxcui:
                best_rxcui = in_override_rxcui
                best_score = score_by_rxcui.get(in_override_rxcui, best_score)
        best_tty_map = concept_ttys.get(best_rxcui, {})
        best_tty = choose_primary_tty(best_tty_map)
        projected = project_ttys(
            start_rxcui=best_rxcui,
            mention_text=mention_text,
            mention_norm=projection_norm,
            mention_context_norm=mention_context_norm,
            conn=conn,
            concept_ttys=concept_ttys,
            max_depth=max_graph_depth,
            candidate_cache=candidate_cache,
            ingredient_cache=ingredient_cache,
        )
        brand_present = has_brand_evidence(
            conn=conn,
            mention_text=mention_text,
            exact_ttys=exact_ttys,
            best_tty=best_tty,
        )
        if not brand_present:
            projected["BN"] = None
            projected["SBD"] = None
            projected["BPCK"] = None

        results.append(
            {
                "mention_text": mention_text,
                "matched_text": matched_text,
                "span": {
                    "start": mention_start,
                    "end": mention_end,
                },
                "matched_span": {
                    "start": match_start,
                    "end": match_end,
                },
                "normalized_text": mention_norm,
                "best_match": {
                    "rxcui": best_rxcui,
                    "name": preferred_names.get(best_rxcui, ""),
                    "tty": best_tty,
                    "score": round(float(best_score), 6),
                    "exact_match": best_rxcui in exact_rxcuids,
                },
                "tty_results": projected,
            }
        )

    return {
        "input_text": input_text,
        "target_ttys": list(TARGET_TTYS),
        "mention_count": len(results),
        "mentions": results,
    }


def cmd_infer(args: argparse.Namespace) -> int:
    index_dir = Path(args.index_dir).expanduser().resolve()
    db_path = index_dir / "rxnorm_index.sqlite"
    emb_path = index_dir / "concept_embeddings.npy"
    rxcui_path = index_dir / "concept_rxcuis.json"

    if args.text is not None:
        input_text = args.text
    else:
        input_text = Path(args.text_file).read_text(encoding="utf-8")

    if not db_path.exists():
        raise FileNotFoundError(f"Missing index sqlite file: {db_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    if not rxcui_path.exists():
        raise FileNotFoundError(f"Missing concept id file: {rxcui_path}")

    with rxcui_path.open("r", encoding="utf-8") as handle:
        rxcui_order = json.load(handle)
    try:
        embeddings = np.load(emb_path, mmap_mode="r")
    except ValueError:
        embeddings = np.load(emb_path)
    if int(embeddings.shape[0]) != len(rxcui_order):
        raise ValueError(
            "Embedding row count does not match concept list size: "
            f"{embeddings.shape[0]} vs {len(rxcui_order)}"
        )

    conn = sqlite3.connect(str(db_path))
    try:
        concept_ttys = load_concept_ttys(conn)
        preferred_names = load_preferred_names(conn)
        output = infer_text_with_resources(
            input_text=input_text,
            conn=conn,
            embeddings=embeddings,
            rxcui_order=rxcui_order,
            concept_ttys=concept_ttys,
            preferred_names=preferred_names,
            top_k=args.top_k,
            exact_boost=args.exact_boost,
            max_graph_depth=args.max_graph_depth,
            max_ngram=args.max_ngram,
            max_exact_candidates=args.max_exact_candidates,
        )
        print(json.dumps(output, indent=2))
        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RxNorm text mapper for SBD/SCD/GPCK/BPCK/BN/SCDC/IN/PIN/MIN."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser(
        "build-index",
        help="Build SQLite + embedding artifacts from RXNCONSO/RXNREL.",
    )
    build_cmd.add_argument(
        "--rrf-dir",
        required=True,
        help="Directory containing RXNCONSO.RRF and RXNREL.RRF.",
    )
    build_cmd.add_argument(
        "--out-dir",
        default="artifacts/rxnorm_index",
        help="Directory to write index artifacts.",
    )
    build_cmd.add_argument(
        "--dims",
        type=int,
        default=256,
        help="Embedding dimensionality for hashed n-gram embeddings.",
    )
    build_cmd.add_argument(
        "--max-aliases-per-concept",
        type=int,
        default=8,
        help="Max normalized aliases used to build each concept embedding.",
    )
    build_cmd.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="Optional cap for concept count (for quick smoke tests).",
    )
    build_cmd.add_argument(
        "--max-rel-lines",
        type=int,
        default=None,
        help="Optional cap for RXNREL lines scanned (for quick smoke tests).",
    )
    build_cmd.set_defaults(func=cmd_build_index)

    infer_cmd = subparsers.add_parser(
        "infer",
        help="Map input text to best RxNorm concepts and project target TTYs.",
    )
    infer_cmd.add_argument(
        "--index-dir",
        default="artifacts/rxnorm_index",
        help="Directory containing index artifacts from build-index.",
    )
    text_group = infer_cmd.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Input text to map.")
    text_group.add_argument("--text-file", help="Read input text from file path.")
    infer_cmd.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Embedding candidates to consider before final ranking.",
    )
    infer_cmd.add_argument(
        "--exact-boost",
        type=float,
        default=0.35,
        help="Score boost for exact alias matches.",
    )
    infer_cmd.add_argument(
        "--max-graph-depth",
        type=int,
        default=3,
        help="Max RxNorm relation hops when projecting target TTYs.",
    )
    infer_cmd.add_argument(
        "--max-ngram",
        type=int,
        default=8,
        help="Max n-gram size for exact mention detection.",
    )
    infer_cmd.add_argument(
        "--max-exact-candidates",
        type=int,
        default=25,
        help="Cap on number of exact-match candidates per mention.",
    )
    infer_cmd.set_defaults(func=cmd_infer)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
