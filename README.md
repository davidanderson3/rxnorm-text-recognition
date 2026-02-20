# RxNorm Text Recognition

This app takes free text, finds medication mentions, picks a best RxNorm concept, and projects to:

- `SBD`
- `SCD`
- `GPCK`
- `BPCK`
- `BN`
- `SCDC`
- `IN`
- `PIN`
- `MIN`

It uses:

- exact alias matching from `RXNCONSO.RRF`
- hashed character n-gram embeddings for semantic-ish retrieval
- relation-constrained traversal over `RXNREL.RRF` for TTY projection
- ingredient-aware filtering to avoid unrelated combination products
- context-aware re-ranking (strength/form cues from nearby text)
- simple negation filtering (for phrases like `denies`, `no`, `discontinued`, `held`)
- brand-aware suppression: if no explicit brand evidence is present in text, `BN`/`SBD`/`BPCK` are left null
- noisy-text recovery for messy inputs (spaced letters, symbol breaks, common shorthand typos)

## Prereqs

- Python 3.9+
- `numpy`

## Build index

```sh
python3 rxnorm_text_recognition.py build-index \
  --rrf-dir RxNorm_full_prescribe_current/rrf \
  --out-dir artifacts/rxnorm_index
```

Artifacts created:

- `artifacts/rxnorm_index/rxnorm_index.sqlite`
- `artifacts/rxnorm_index/concept_embeddings.npy`
- `artifacts/rxnorm_index/concept_rxcuis.json`
- `artifacts/rxnorm_index/metadata.json`

## Run inference

```sh
python3 rxnorm_text_recognition.py infer \
  --index-dir artifacts/rxnorm_index \
  --text "Patient takes metformin 500 mg BID and lisinopril 10 mg daily."
```

Output is JSON with:

- `mention_text`: full line-level medication text context
- `matched_text`: exact phrase span matched/retrieved
- `span` and `matched_span`
- best match (`rxcui`, `name`, `tty`, score)
- projected results for all target TTYs
- when a `MIN` is found, `tty_results.IN` contains all ingredient `IN` entries

## Run the web interface (local)

```sh
python3 rxnorm_web.py --index-dir artifacts/rxnorm_index --port 8000
```

Then open:

```txt
http://127.0.0.1:8000
```

The page now runs inference in-browser via Pyodide (no `/api/infer` call needed).

## Quick smoke test (small subset)

Use these flags to test quickly before full indexing:

```sh
python3 rxnorm_text_recognition.py build-index \
  --rrf-dir RxNorm_full_prescribe_current/rrf \
  --out-dir artifacts/rxnorm_index_smoke \
  --max-concepts 5000 \
  --max-rel-lines 200000
```
