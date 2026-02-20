const DEFAULT_CONFIG = {
  pyodideIndexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.3/full/",
  artifactsBaseUrl: "../artifacts/rxnorm_index",
  scriptUrl: "../rxnorm_text_recognition.py",
  inferOptions: {
    top_k: 40,
    exact_boost: 0.35,
    max_graph_depth: 3,
    max_ngram: 8,
    max_exact_candidates: 25,
  },
};

const userConfig = window.RXNORM_WEB_CONFIG || {};
const CONFIG = {
  ...DEFAULT_CONFIG,
  ...userConfig,
  inferOptions: {
    ...DEFAULT_CONFIG.inferOptions,
    ...(userConfig.inferOptions || {}),
  },
};

const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const inputText = document.getElementById("inputText");
const statusEl = document.getElementById("status");
const resultList = document.getElementById("resultList");
const rawJson = document.getElementById("rawJson");

let pyodidePromise = null;
let enginePromise = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? "status error" : "status";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function cellValue(item) {
  if (Array.isArray(item)) {
    return item.map((entry) => cellValue(entry)).filter(Boolean).join(" | ");
  }
  if (!item) return "-";
  return `${item.name} (${item.rxcui})`;
}

function ttyIdentity(item) {
  if (!item || typeof item !== "object") return "";
  if (item.rxcui) return String(item.rxcui);
  if (item.name) return String(item.name);
  return "";
}

function mergeTTYLists(current, candidate) {
  const merged = [];
  const seen = new Set();
  for (const source of [current || [], candidate || []]) {
    if (!Array.isArray(source)) continue;
    for (const item of source) {
      if (!item) continue;
      const id = ttyIdentity(item);
      if (!id || seen.has(id)) continue;
      seen.add(id);
      merged.push(item);
    }
  }
  return merged;
}

function pickBetterTTY(current, candidate) {
  if (!current) return candidate || null;
  if (!candidate) return current;
  if (Array.isArray(current) || Array.isArray(candidate)) {
    return mergeTTYLists(current, candidate);
  }
  const currentDepth =
    typeof current.depth === "number" ? current.depth : Number.POSITIVE_INFINITY;
  const candidateDepth =
    typeof candidate.depth === "number" ? candidate.depth : Number.POSITIVE_INFINITY;
  if (candidateDepth < currentDepth) return candidate;
  return current;
}

function mergeMentionRows(base, incoming) {
  const merged = {
    ...base,
    tty_results: { ...(base.tty_results || {}) },
  };
  const incomingTTY = incoming.tty_results || {};
  const ttyKeys = ["SBD", "SCD", "GPCK", "BPCK", "BN", "SCDC", "IN", "PIN", "MIN"];
  for (const key of ttyKeys) {
    merged.tty_results[key] = pickBetterTTY(
      merged.tty_results[key] || null,
      incomingTTY[key] || null,
    );
  }
  merged.tty_results.IN_ALL = mergeTTYLists(
    merged.tty_results.IN_ALL,
    incomingTTY.IN_ALL,
  );
  if (merged.tty_results.IN_ALL.length) {
    const inCurrent = merged.tty_results.IN;
    const inAsList = Array.isArray(inCurrent) ? inCurrent : inCurrent ? [inCurrent] : [];
    merged.tty_results.IN_ALL = mergeTTYLists(merged.tty_results.IN_ALL, inAsList);
    merged.tty_results.IN = mergeTTYLists(inAsList, merged.tty_results.IN_ALL);
  } else if (Array.isArray(merged.tty_results.IN)) {
    merged.tty_results.IN_ALL = mergeTTYLists(merged.tty_results.IN, []);
  }

  const baseNorm = String(merged.normalized_text || "").trim();
  const incomingNorm = String(incoming.normalized_text || "").trim();
  if (incomingNorm && incomingNorm !== baseNorm) {
    const parts = baseNorm
      ? baseNorm.split(" | ").map((p) => p.trim()).filter(Boolean)
      : [];
    if (!parts.includes(incomingNorm)) {
      parts.push(incomingNorm);
    }
    merged.normalized_text = parts.join(" | ");
  }
  return merged;
}

function dedupeMentionsForDisplay(mentions) {
  const deduped = [];
  const indexByKey = new Map();
  for (const mention of mentions) {
    const scdRxcui = mention.tty_results && mention.tty_results.SCD
      ? String(mention.tty_results.SCD.rxcui || "")
      : "";
    const mentionKey = String(mention.mention_text || "").trim().toLowerCase();
    if (!scdRxcui || !mentionKey) {
      deduped.push(mention);
      continue;
    }

    const key = `${mentionKey}::${scdRxcui}`;
    const existingIdx = indexByKey.get(key);
    if (existingIdx === undefined) {
      indexByKey.set(key, deduped.length);
      deduped.push(mention);
      continue;
    }

    deduped[existingIdx] = mergeMentionRows(deduped[existingIdx], mention);
  }
  return deduped;
}

function ttyLine(label, item) {
  if (Array.isArray(item) && item.length === 0) return "";
  if (!item) return "";
  return `<div class="tty-line"><span class="tty-label">${label}</span> ${escapeHtml(cellValue(item))}</div>`;
}

function renderResults(payload) {
  const mentions = payload.mentions || [];
  const displayMentions = dedupeMentionsForDisplay(mentions);
  const ttyOrder = ["SCD", "SBD", "GPCK", "BPCK", "BN", "SCDC", "IN", "PIN", "MIN"];
  if (!displayMentions.length) {
    resultList.innerHTML = `<article class="result-item empty">No medication mentions mapped.</article>`;
  } else {
    resultList.innerHTML = displayMentions
      .map((m) => {
        const t = m.tty_results || {};
        const ttyRows = ttyOrder
          .map((label) => {
            const value =
              label === "IN" && Array.isArray(t.IN_ALL) && t.IN_ALL.length
                ? t.IN_ALL
                : t[label];
            return ttyLine(label, value);
          })
          .filter(Boolean)
          .join("");
        const ttySection = ttyRows ? `<div class="tty-list">${ttyRows}</div>` : "";
        return `
          <article class="result-item">
            <strong>${escapeHtml(m.mention_text || "")}</strong>
            <small>${escapeHtml(m.normalized_text || "")}</small>
            ${ttySection}
          </article>
        `;
      })
      .join("");
  }

  rawJson.textContent = JSON.stringify(payload, null, 2);
}

function clearOutput() {
  resultList.innerHTML = `<article class="result-item empty">No results yet.</article>`;
  rawJson.textContent = "{}";
  setStatus("Cleared.");
}

function joinUrl(base, path) {
  return `${String(base).replace(/\/+$/, "")}/${String(path).replace(/^\/+/, "")}`;
}

async function fetchText(url, label) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${label}: ${res.status} ${res.statusText}`);
  }
  return await res.text();
}

async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${label}: ${res.status} ${res.statusText}`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

async function ensurePyodideScript() {
  if (typeof window.loadPyodide === "function") {
    return;
  }
  const scriptUrl = joinUrl(CONFIG.pyodideIndexURL, "pyodide.js");
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = scriptUrl;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${scriptUrl}`));
    document.head.appendChild(script);
  });
}

async function getPyodide() {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      setStatus("Loading Python runtime...");
      await ensurePyodideScript();
      return await window.loadPyodide({ indexURL: CONFIG.pyodideIndexURL });
    })().catch((err) => {
      pyodidePromise = null;
      throw err;
    });
  }
  return pyodidePromise;
}

async function initializeEngine() {
  const py = await getPyodide();
  setStatus("Loading Python packages...");
  await py.loadPackage(["numpy", "sqlite3"]);
  py.FS.mkdirTree("/rxnorm");
  py.FS.mkdirTree("/rxnorm/index");

  setStatus("Downloading inference script...");
  const engineScript = await fetchText(CONFIG.scriptUrl, "rxnorm_text_recognition.py");
  py.FS.writeFile("/rxnorm/rxnorm_text_recognition.py", engineScript);

  setStatus("Downloading SQLite index (this can take a while)...");
  let sqliteBytes = await fetchBytes(
    joinUrl(CONFIG.artifactsBaseUrl, "rxnorm_index.sqlite"),
    "rxnorm_index.sqlite",
  );
  py.FS.writeFile("/rxnorm/index/rxnorm_index.sqlite", sqliteBytes);
  sqliteBytes = null;

  setStatus("Downloading concept embeddings (this can take a while)...");
  let embBytes = await fetchBytes(
    joinUrl(CONFIG.artifactsBaseUrl, "concept_embeddings.npy"),
    "concept_embeddings.npy",
  );
  py.FS.writeFile("/rxnorm/index/concept_embeddings.npy", embBytes);
  embBytes = null;

  setStatus("Downloading concept IDs...");
  const rxcuiJson = await fetchText(
    joinUrl(CONFIG.artifactsBaseUrl, "concept_rxcuis.json"),
    "concept_rxcuis.json",
  );
  py.FS.writeFile("/rxnorm/index/concept_rxcuis.json", rxcuiJson);

  setStatus("Initializing inference engine...");
  await py.runPythonAsync(`
import importlib.util
import json
import sys
import sqlite3
import numpy as np

_spec = importlib.util.spec_from_file_location("rxnorm_engine", "/rxnorm/rxnorm_text_recognition.py")
rxnorm_engine = importlib.util.module_from_spec(_spec)
sys.modules["rxnorm_engine"] = rxnorm_engine
_spec.loader.exec_module(rxnorm_engine)

_WEB_CONN = sqlite3.connect("/rxnorm/index/rxnorm_index.sqlite")
with open("/rxnorm/index/concept_rxcuis.json", "r", encoding="utf-8") as _handle:
    _WEB_RXCUI_ORDER = json.load(_handle)
_WEB_EMBEDDINGS = np.load("/rxnorm/index/concept_embeddings.npy")
_WEB_CONCEPT_TTYS = rxnorm_engine.load_concept_ttys(_WEB_CONN)
_WEB_PREFERRED_NAMES = rxnorm_engine.load_preferred_names(_WEB_CONN)
_WEB_CANDIDATE_CACHE = {}
_WEB_INGREDIENT_CACHE = {}
`);
  return py;
}

async function ensureEngine() {
  if (!enginePromise) {
    enginePromise = initializeEngine().catch((err) => {
      enginePromise = null;
      throw err;
    });
  }
  return await enginePromise;
}

function runPythonInference(py, text) {
  const opts = CONFIG.inferOptions;
  py.globals.set("WEB_INPUT_TEXT", text);
  return py.runPythonAsync(`
import json
_WEB_RESULT = rxnorm_engine.infer_text_with_resources(
    input_text=WEB_INPUT_TEXT,
    conn=_WEB_CONN,
    embeddings=_WEB_EMBEDDINGS,
    rxcui_order=_WEB_RXCUI_ORDER,
    concept_ttys=_WEB_CONCEPT_TTYS,
    preferred_names=_WEB_PREFERRED_NAMES,
    top_k=${Number(opts.top_k)},
    exact_boost=${Number(opts.exact_boost)},
    max_graph_depth=${Number(opts.max_graph_depth)},
    max_ngram=${Number(opts.max_ngram)},
    max_exact_candidates=${Number(opts.max_exact_candidates)},
    candidate_cache=_WEB_CANDIDATE_CACHE,
    ingredient_cache=_WEB_INGREDIENT_CACHE,
)
json.dumps(_WEB_RESULT)
`);
}

async function runInference() {
  const text = inputText.value.trim();
  if (!text) {
    setStatus("Enter text first.", true);
    return;
  }

  runBtn.disabled = true;
  try {
    const py = await ensureEngine();
    if (!py || !py.globals) {
      throw new Error("Inference engine failed to initialize.");
    }
    setStatus("Running inference...");
    const payloadText = await runPythonInference(py, text);
    try {
      py.globals.delete("WEB_INPUT_TEXT");
    } catch (_err) {
      // Ignore best-effort cleanup errors.
    }
    const payload = JSON.parse(payloadText);
    renderResults(payload);
    setStatus("Done.");
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    setStatus(`Inference failed: ${message}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runInference);
clearBtn.addEventListener("click", clearOutput);
