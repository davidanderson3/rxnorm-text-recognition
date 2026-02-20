#!/usr/bin/env python3
"""RxNorm MVP: build an index from RRF files and map free text to target TTYs."""

from __future__ import annotations

import argparse
import hashlib
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
STRENGTH_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|meq|%)\b")

FORM_HINTS: Dict[str, Tuple[str, ...]] = {
    "inhaler": ("inhaler", "inhalation", "hfa", "mdi", "actuat"),
    "tablet": ("tablet", "tab", "oral tablet"),
    "capsule": ("capsule", "cap"),
    "solution": ("solution", "soln"),
    "suspension": ("suspension",),
    "injection": ("inject", "intravenous", "subcutaneous", "intramuscular"),
    "cream": ("cream",),
    "ointment": ("ointment",),
    "patch": ("patch",),
    "nasal": ("nasal",),
    "ophthalmic": ("ophthalmic", "eye"),
    "otologic": ("otic", "ear"),
    "er": ("extended release", "er", "xr", "24 hr"),
    "ec": ("enteric", "ec"),
}

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
            norm_span = normalize_text(span_text)
            if not norm_span:
                i += 1
                continue

            rows = conn.execute(
                "SELECT DISTINCT rxcui, tty FROM alias WHERE norm_text = ? LIMIT ?",
                (norm_span, max_exact_candidates),
            ).fetchall()
            if not rows:
                i += 1
                continue

            for j in range(i, i + n):
                occupied[j] = True

            mentions.append(
                {
                    "text": span_text,
                    "norm_text": norm_span,
                    "start": start,
                    "end": end,
                    "exact_rxcuids": [row[0] for row in rows],
                    "exact_ttys": sorted({row[1] for row in rows if row[1]}),
                }
            )
            i += n

    mentions.sort(key=lambda item: int(item["start"]))
    if mentions:
        line_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        line_boundaries: Dict[int, Tuple[int, int]] = {}
        for idx, mention in enumerate(mentions):
            span_start = int(mention["start"])
            span_end = int(mention["end"])
            line_start = text.rfind("\n", 0, span_start)
            line_start = 0 if line_start < 0 else line_start + 1
            line_end = text.find("\n", span_end)
            line_end = len(text) if line_end < 0 else line_end
            key = (line_start, line_end)
            line_boundaries[idx] = key
            line_counts[key] += 1

        for idx, mention in enumerate(mentions):
            line_start, line_end = line_boundaries[idx]
            line_raw = text[line_start:line_end]
            line_stripped = line_raw.strip()
            if not line_stripped:
                mention["mention_text"] = str(mention["text"])
                mention["mention_start"] = int(mention["start"])
                mention["mention_end"] = int(mention["end"])
                continue

            if line_counts[(line_start, line_end)] == 1:
                left_trim = len(line_raw) - len(line_raw.lstrip())
                mention["mention_text"] = line_stripped
                mention["mention_start"] = line_start + left_trim
                mention["mention_end"] = line_start + left_trim + len(line_stripped)
            else:
                mention["mention_text"] = str(mention["text"])
                mention["mention_start"] = int(mention["start"])
                mention["mention_end"] = int(mention["end"])
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
                    "mention_text": chunk,
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
            "mention_text": stripped,
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


def extract_mention_features(
    mention_text: str, mention_norm: str, context_norm: str
) -> Dict[str, object]:
    strength_tokens = {m.group(0).strip() for m in STRENGTH_RE.finditer(context_norm)}
    form_tokens: Set[str] = set()
    padded_norm = f" {context_norm} "
    for form_key, hints in FORM_HINTS.items():
        for hint in hints:
            if f" {hint} " in padded_norm:
                form_tokens.add(form_key)
                break

    combo_hint = (
        "/" in mention_text
        or "+" in mention_text
        or " and " in f" {mention_norm} "
        or " with " in f" {mention_norm} "
    )
    return {
        "strength_tokens": strength_tokens,
        "form_tokens": form_tokens,
        "combo_hint": combo_hint,
    }


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
) -> float:
    candidate_norm = normalize_text(candidate_name)
    lexical = token_jaccard(mention_norm, candidate_norm)
    score = 1.8 * lexical
    score -= 0.45 * float(depth)

    strength_tokens = mention_features["strength_tokens"]  # type: ignore[assignment]
    form_tokens = mention_features["form_tokens"]  # type: ignore[assignment]
    combo_hint = bool(mention_features["combo_hint"])

    if isinstance(strength_tokens, set) and strength_tokens:
        matched = sum(1 for token in strength_tokens if token in candidate_norm)
        if matched > 0:
            score += 1.6 * float(matched)
        else:
            score -= 0.45

    if isinstance(form_tokens, set) and form_tokens:
        matched_form = 0
        for form_key in form_tokens:
            for hint in FORM_HINTS.get(str(form_key), ()):
                if hint in candidate_norm:
                    matched_form += 1
                    break
        if matched_form > 0:
            score += 0.7 * float(matched_form)
        else:
            score -= 0.25

    combo_allowed = combo_hint or len(anchor_ingredients) > 1
    if target_tty == "MIN" and not combo_allowed:
        return -1000.0

    if anchor_ingredients:
        if candidate_ingredients:
            if candidate_ingredients == anchor_ingredients:
                score += 2.4
            elif candidate_ingredients.issuperset(anchor_ingredients):
                score += 0.3
                if not combo_allowed:
                    score -= 2.6
            elif candidate_ingredients & anchor_ingredients:
                score += 0.1
                if not combo_allowed and len(candidate_ingredients) > len(anchor_ingredients):
                    score -= 2.4
            else:
                score -= 3.0
        else:
            score -= 0.4

    if not combo_allowed and "/" in candidate_norm and target_tty in {
        "SBD",
        "SCD",
        "SCDC",
        "BN",
        "GPCK",
        "BPCK",
    }:
        score -= 2.0

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
) -> Dict[str, Optional[Dict[str, object]]]:
    projected: Dict[str, Optional[Dict[str, object]]] = {tty: None for tty in TARGET_TTYS}
    mention_features = extract_mention_features(
        mention_text, mention_norm, mention_context_norm
    )
    anchor_ingredients = ingredient_set_for_rxcui(
        start_rxcui, conn, concept_ttys, ingredient_cache
    )

    for target_tty in TARGET_TTYS:
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

    return projected


def load_concept_ttys(conn: sqlite3.Connection) -> Dict[str, Dict[str, str]]:
    concept_ttys: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in conn.execute("SELECT rxcui, tty, name FROM concept_tty"):
        concept_ttys[row[0]][row[1]] = row[2]
    return concept_ttys


def load_preferred_names(conn: sqlite3.Connection) -> Dict[str, str]:
    return {row[0]: row[1] for row in conn.execute("SELECT rxcui, preferred_name FROM concepts")}


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
    embeddings = np.load(emb_path, mmap_mode="r")
    if int(embeddings.shape[0]) != len(rxcui_order):
        raise ValueError(
            "Embedding row count does not match concept list size: "
            f"{embeddings.shape[0]} vs {len(rxcui_order)}"
        )

    conn = sqlite3.connect(str(db_path))
    try:
        concept_ttys = load_concept_ttys(conn)
        preferred_names = load_preferred_names(conn)

        mentions = detect_mentions(
            text=input_text,
            conn=conn,
            max_ngram=args.max_ngram,
            max_exact_candidates=args.max_exact_candidates,
        )
        results: List[Dict[str, object]] = []
        candidate_cache: Dict[Tuple[str, str, int], List[Tuple[str, int]]] = {}
        ingredient_cache: Dict[str, Set[str]] = {}

        for mention in mentions:
            mention_text = str(mention["text"])
            mention_norm = str(mention["norm_text"])
            mention_start = int(mention["start"])
            mention_end = int(mention["end"])
            line_start = input_text.rfind("\n", 0, mention_start)
            line_start = 0 if line_start < 0 else line_start + 1
            line_end = input_text.find("\n", mention_end)
            line_end = len(input_text) if line_end < 0 else line_end
            line_context = input_text[line_start:line_end].strip()
            if len(line_context) < 4:
                context_start = max(0, mention_start - 24)
                context_end = min(len(input_text), mention_end + 40)
                line_context = input_text[context_start:context_end]
            mention_context_norm = normalize_text(line_context)
            exact_rxcuids: List[str] = list(mention["exact_rxcuids"])  # type: ignore[assignment]
            score_by_rxcui: Dict[str, float] = {}

            for rxcui, score in top_embedding_candidates(
                mention_norm=mention_norm,
                embeddings=embeddings,
                rxcui_order=rxcui_order,
                top_k=args.top_k,
            ):
                score_by_rxcui[rxcui] = score

            for rxcui in exact_rxcuids:
                score_by_rxcui[rxcui] = score_by_rxcui.get(rxcui, 0.0) + args.exact_boost

            if not score_by_rxcui:
                continue

            best_rxcui, best_score = max(score_by_rxcui.items(), key=lambda item: item[1])
            best_tty_map = concept_ttys.get(best_rxcui, {})
            best_tty = choose_primary_tty(best_tty_map)
            projected = project_ttys(
                start_rxcui=best_rxcui,
                mention_text=mention_text,
                mention_norm=mention_norm,
                mention_context_norm=mention_context_norm,
                conn=conn,
                concept_ttys=concept_ttys,
                max_depth=args.max_graph_depth,
                candidate_cache=candidate_cache,
                ingredient_cache=ingredient_cache,
            )

            results.append(
                {
                    "mention_text": mention_text,
                    "span": {
                        "start": mention_start,
                        "end": mention_end,
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

        output = {
            "input_text": input_text,
            "target_ttys": list(TARGET_TTYS),
            "mention_count": len(results),
            "mentions": results,
        }
        print(json.dumps(output, indent=2))
        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RxNorm MVP text mapper for SBD/SCD/GPCK/BPCK/BN/SCDC/IN/PIN/MIN."
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
        default="artifacts/rxnorm_mvp",
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
        default="artifacts/rxnorm_mvp",
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
