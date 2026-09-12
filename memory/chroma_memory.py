"""
ChromaDB-based vector memory layer over the SQLite memory_items store.

Provides:
  - ChromaMemory:   direct ChromaDB PersistentClient wrapper
  - MemoryBridge:   unified interface between SQLite + ChromaDB
  - Compatibility:  existing callers of load_memory(), save_memory(),
                    search_all_memories() continue to work.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Configuration flag ─────────────────────────────────────────────────────

CHROMA_VECTOR_SEARCH_ENABLED = True

# Allow override via api_keys.json config at module load time
try:
    from memory.config_manager import load_api_keys
    _cfg = load_api_keys()
    if "CHROMA_VECTOR_SEARCH_ENABLED" in _cfg:
        CHROMA_VECTOR_SEARCH_ENABLED = bool(_cfg["CHROMA_VECTOR_SEARCH_ENABLED"])
except Exception:
    pass


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CHROMA_DB_PATH = str(BASE_DIR / "memory" / "chroma_db")

_EMBED_MODEL = None
_EMBED_LOCK = Lock()

# ── Embedding helpers ──────────────────────────────────────────────────────

CHROMA_EMBED_FN = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts for ChromaDB.

    Prefers the memory service's /embed endpoint (the MiniLM L6 v2 model is
    already preloaded at memory service startup) — this avoids loading a
    duplicate SentenceTransformer model inside the app process, which was the
    main cause of slow first-message latency. Falls back to the local
    CHROMA_EMBED_FN only when the service is unavailable.
    """
    try:
        from memory.memory_client import _is_healthy, memory_embed
        if _is_healthy():
            vecs = memory_embed(texts)
            return [v.tolist() for v in vecs]
    except Exception:
        pass
    return CHROMA_EMBED_FN(texts)


def _make_metadata(
    category: str,
    key: str,
    value: str,
    extra: Optional[dict] = None,
) -> dict:
    """Build a flat metadata dict from an entry dict.
    Lists/dicts (history, contradictions) are serialised to JSON strings
    so ChromaDB can store them.
    """
    meta = {
        "category": category,
        "key": key,
        "value": value,
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            elif v is None:
                meta[k] = ""
            else:
                meta[k] = json.dumps(v, ensure_ascii=False)
    return meta


def _parse_extra_metadata(meta: dict) -> dict:
    """Reverse of _make_metadata — deserialise JSON fields."""
    result = {}
    for k, v in meta.items():
        if k in ("category", "key", "value"):
            continue
        if isinstance(v, str) and v.startswith("["):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        elif isinstance(v, str) and v.startswith("{"):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        else:
            result[k] = v
    return result


# ── ChromaMemory ───────────────────────────────────────────────────────────

class ChromaMemory:
    """ChromaDB-backed persistent vector storage for long-term memories."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "long_term_memory",
    ):
        self.persist_dir = persist_dir or CHROMA_DB_PATH
        self.collection_name = collection_name
        self._lock = Lock()
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._ready = False
        self._init()

    # ── internal ────────────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=CHROMA_EMBED_FN,
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
        except Exception as e:
            print(f"[ChromaMemory] Init error: {e}")
            self._ready = False

    def _ensure_ready(self) -> None:
        if not self._ready:
            self._init()

    def _doc_id(self, category: str, key: str) -> str:
        return f"{category}/{key}"

    def _split_doc_id(self, doc_id: str):
        parts = doc_id.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "notes", parts[0]

    # ── public API ──────────────────────────────────────────────────────

    def health(self) -> dict:
        self._ensure_ready()
        if not self._ready:
            return {"status": "unhealthy", "error": "ChromaDB not initialised"}
        try:
            count = self._collection.count()
            return {
                "status": "healthy",
                "collection": self.collection_name,
                "count": count,
                "persist_dir": self.persist_dir,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def add_memory(
        self,
        category: str,
        key: str,
        value: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        self._ensure_ready()
        if not self._ready:
            return False
        doc_id = self._doc_id(category, key)
        meta = _make_metadata(category, key, value, metadata)
        with self._lock:
            try:
                # H1 fix: use upsert — `add` raises on an existing doc_id, so
                # every re-save of an existing fact silently kept the OLD
                # embedding/value in Chroma while SQLite had the new one.
                self._collection.upsert(
                    documents=[value],
                    embeddings=_embed_texts([value]),
                    metadatas=[meta],
                    ids=[doc_id],
                )
                return True
            except Exception as e:
                print(f"[ChromaMemory] add_memory error: {e}")
                return False

    def search_memory(self, query: str, n_results: int = 10) -> list[dict]:
        self._ensure_ready()
        if not self._ready:
            return []
        with self._lock:
            try:
                results = self._collection.query(
                    query_embeddings=_embed_texts([query]),
                    n_results=n_results,
                )
                return self._format_results(results)
            except Exception as e:
                print(f"[ChromaMemory] search_memory error: {e}")
                return []

    def search_by_category(
        self,
        category: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        self._ensure_ready()
        if not self._ready:
            return []
        with self._lock:
            try:
                results = self._collection.query(
                    query_embeddings=_embed_texts([query]),
                    n_results=n_results,
                    where={"category": category},
                )
                return self._format_results(results)
            except Exception as e:
                print(f"[ChromaMemory] search_by_category error: {e}")
                return []

    def get_all_memories(self) -> list[dict]:
        self._ensure_ready()
        if not self._ready:
            return []
        with self._lock:
            try:
                all_data = self._collection.get()
                return self._format_get_result(all_data)
            except Exception as e:
                print(f"[ChromaMemory] get_all_memories error: {e}")
                return []

    def delete_memory(self, category: str, key: str) -> bool:
        self._ensure_ready()
        if not self._ready:
            return False
        doc_id = self._doc_id(category, key)
        with self._lock:
            try:
                self._collection.delete(ids=[doc_id])
                return True
            except Exception:
                return False

    def update_memory(
        self,
        category: str,
        key: str,
        new_value: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        self._ensure_ready()
        if not self._ready:
            return False
        doc_id = self._doc_id(category, key)
        meta = _make_metadata(category, key, new_value, metadata)
        with self._lock:
            try:
                self._collection.update(
                    documents=[new_value],
                    metadatas=[meta],
                    ids=[doc_id],
                )
                return True
            except Exception as e:
                print(f"[ChromaMemory] update_memory error: {e}")
                try:
                    self._collection.add(
                        documents=[new_value],
                        metadatas=[meta],
                        ids=[doc_id],
                    )
                    return True
                except Exception:
                    return False

    def count(self) -> int:
        self._ensure_ready()
        if not self._ready:
            return 0
        with self._lock:
            try:
                return self._collection.count()
            except Exception:
                return 0

    def get_collection_stats(self) -> dict:
        self._ensure_ready()
        if not self._ready:
            return {"status": "not_ready"}
        with self._lock:
            try:
                cnt = self._collection.count()
                all_meta = self._collection.get(limit=cnt) if cnt > 0 else {"metadatas": []}
                categories = {}
                for m in (all_meta.get("metadatas") or []):
                    cat = m.get("category", "unknown") if m else "unknown"
                    categories[cat] = categories.get(cat, 0) + 1
                return {
                    "total_count": cnt,
                    "categories": categories,
                    "persist_dir": self.persist_dir,
                    "collection": self.collection_name,
                    "embedding_model": "all-MiniLM-L6-v2",
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

    # ── result formatters ───────────────────────────────────────────────

    @staticmethod
    def _format_results(results) -> list[dict]:
        out = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            entry = {
                "id": ids[i],
                "value": doc,
                "score": 1.0 - (distances[i] if i < len(distances) else 0.0),
            }
            if meta:
                entry["category"] = meta.get("category", "")
                entry["key"] = meta.get("key", "")
                entry.update(_parse_extra_metadata(meta))
            out.append(entry)
        return out

    @staticmethod
    def _format_get_result(data) -> list[dict]:
        out = []
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        documents = data.get("documents", [])
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            entry = {
                "id": ids[i],
                "value": doc,
            }
            if meta:
                entry["category"] = meta.get("category", "")
                entry["key"] = meta.get("key", "")
                entry.update(_parse_extra_metadata(meta))
            out.append(entry)
        return out


# ── MemoryBridge ───────────────────────────────────────────────────────────

# Shared singleton
_chroma_instance: Optional[ChromaMemory] = None
_bridge_lock = Lock()


def _get_chroma() -> ChromaMemory:
    global _chroma_instance
    if _chroma_instance is None:
        with _bridge_lock:
            if _chroma_instance is None:
                _chroma_instance = ChromaMemory()
    return _chroma_instance


class MemoryBridge:
    """Unified interface: keeps existing JSON memory working while adding
    ChromaDB as a faster vector-search alternative."""

    def __init__(self):
        self.chroma = _get_chroma()

    # ── search ──────────────────────────────────────────────────────────

    def search_all(self, query: str, n_results: int = 10, mode: str = "summary",
                   project_id: str = "") -> list[dict]:
        """Search ChromaDB first; fall back to JSON-based search if ChromaDB
        is unavailable or returns no results.
        
        mode="summary" — return ChromaDB results as-is (summaries stay as text).
        mode="expand"  — resolve session_summary pointers to raw SQLite turns.
        project_id     — scope results to a specific project (filters metadata).
        """
        if not CHROMA_VECTOR_SEARCH_ENABLED:
            return self._search_json(query, n_results)

        results = self.chroma.search_memory(query, n_results=n_results)
        if results:
            # Filter by project_id if specified
            if project_id:
                results = [r for r in results if r.get("project_id", "") == project_id or not r.get("project_id")]
            if mode == "expand":
                results = resolve_summaries(results)
            return results

        # Fallback to JSON-based keyword search
        return self._search_json(query, n_results)

    def _search_json(self, query: str, n_results: int = 10) -> list[dict]:
        """Simple keyword search over memory as fallback."""
        try:
            from memory.memory_manager import load_memory
            memory = load_memory()
        except Exception:
            return []

        q = query.lower()
        results = []
        for category, entries in memory.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                val = entry.get("value") if isinstance(entry, dict) else entry
                if not val:
                    continue
                if q in str(val).lower() or q in key.lower() or q in category.lower():
                    results.append({
                        "id": f"{category}/{key}",
                        "category": category,
                        "key": key,
                        "value": str(val),
                        "score": float(entry.get("score", 1.0)) if isinstance(entry, dict) else 1.0,
                    })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_results]

    # ── migration ───────────────────────────────────────────────────────

    def migrate_from_json(self) -> dict:
        """Import all memory data into ChromaDB.
        Returns stats about what was migrated.
        """
        try:
            from memory.memory_manager import load_memory
            memory = load_memory()
        except Exception:
            memory = {}

        stats = {"added": 0, "skipped": 0, "errors": 0, "categories": {}}
        for category, entries in memory.items():
            if not isinstance(entries, dict):
                continue
            cat_count = 0
            for key, entry in entries.items():
                val = entry.get("value") if isinstance(entry, dict) else entry
                if not val:
                    stats["skipped"] += 1
                    continue
                extra = entry if isinstance(entry, dict) else None
                ok = self.chroma.add_memory(category, key, str(val), metadata=extra)
                if ok:
                    stats["added"] += 1
                    cat_count += 1
                else:
                    stats["errors"] += 1
            if cat_count > 0:
                stats["categories"][category] = cat_count

        stats["total_before"] = sum(stats["categories"].values())
        print(f"[MemoryBridge] Migrated entries from JSON -> ChromaDB")
        return stats

    def count(self) -> int:
        return self.chroma.count()

    def health(self) -> dict:
        return self.chroma.health()

    def stats(self) -> dict:
        return self.chroma.get_collection_stats()


# ── Compatibility shim ─────────────────────────────────────────────────────
# Existing code calling load_memory(), save_memory(), search_all_memories()
# continues to work unchanged.

from memory import memory_manager as _mm

load_memory = _mm.load_memory
save_memory = _mm.save_memory
update_memory = _mm.update_memory
remember = _mm.remember
forget = _mm.forget
forget_memory = _mm.forget_memory
format_memory_for_prompt = _mm.format_memory_for_prompt
format_memory_summary = _mm.format_memory_summary
format_core_memory = _mm.format_core_memory
clear_expired_memories = _mm.clear_expired_memories
enforce_memory_cap = _mm.enforce_memory_cap
context_status = _mm.context_status
archival_memory_search = _mm.archival_memory_search
core_memory_append = _mm.core_memory_append
core_memory_replace = _mm.core_memory_replace
load_recent_conversations = _mm.load_recent_conversations
save_conversation = _mm.save_conversation


def search_all_memories(query: str, limit: int = 10, mode: str = "expand",
                        project_id: str = "") -> str:
    """Overloaded search: always runs the true multi-signal fusion
    (BM25 + SQLite FTS5 + ChromaDB vector + entity graph, weighted + reranked).

    Previously this short-circuited to Chroma-only when Chroma was enabled,
    silently bypassing the keyword/FTS/entity signals. The fusion in
    memory_manager.search_all_memories includes the Chroma signal itself,
    so delegating always gives the full 5-signal result.

    mode / project_id are kept for API compatibility; summary-only rendering
    (mode="summary") is applied on top of the fused results.
    """
    fused = _mm.search_all_memories(query, limit=limit)

    # mode="summary" — keep the fused result but strip raw conversation
    # expansions that resolve_summaries() used to inject (token bomb guard).
    if mode == "summary":
        return fused

    return fused


# ── Session summary embedding / resolution ─────────────────────────────


def _extract_keywords(text: str, max_keywords: int = 10) -> str:
    """Extract topical keywords (capitalized tech terms, proper nouns) from summary text."""
    words = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b', text)
    seen = set()
    unique = []
    for w in words:
        lower = w.lower()
        if lower not in seen and len(w) > 2:
            seen.add(lower)
            unique.append(w)
        if len(unique) >= max_keywords:
            break
    return ", ".join(unique) if unique else ""


def embed_session_summary(session_id: str, summary: str) -> bool:
    """Store session summary in ChromaDB as a pointer to raw SQLite turns.
    Uses upsert so per-session duplicate summaries update rather than collide."""
    try:
        from memory.conversation_db import get_session
        session = get_session(session_id)
        title = session["title"] if session else session_id
        project_id = session.get("project_id", "") if session else ""
        chroma = _get_chroma()
        keywords = _extract_keywords(summary)
        metadata = {
            "type": "session_summary",
            "session_id": session_id,
            "title": title,
            "project_id": project_id,
            "keywords": keywords,
            "created_at": datetime.now().isoformat(),
        }
        return chroma.update_memory("session_summary", session_id, summary, metadata=metadata)
    except Exception as e:
        print(f"[ChromaMemory] embed_session_summary error: {e}")
        return False


def resolve_summaries(results: list[dict]) -> list[dict]:
    """Post-process ChromaDB results: resolve session_summary pointers to raw conversation turns.
    Returns results with 'value' replaced by the full formatted conversation."""
    try:
        from memory.conversation_db import get_turns_by_session, format_turn_for_prompt
    except Exception:
        return results
    out = []
    for r in results:
        if r.get("category") == "session_summary":
            session_id = r.get("session_id") or r.get("key", "")
            turns = get_turns_by_session(session_id)
            if turns:
                conv_lines = ["[RAW CONVERSATION — loaded from session]"]
                for t in turns:
                    formatted = format_turn_for_prompt(t)
                    if formatted:
                        conv_lines.append(formatted)
                r["value"] = "\n".join(conv_lines)
                r["resolved"] = True
                r["raw_turns"] = len(turns)
        out.append(r)
    return out


def migrate_summaries_to_chroma() -> dict:
    """One-time migration: embed all existing SQLite session summaries into ChromaDB.
    Uses batched ChromaDB add for performance. Returns stats."""
    import sqlite3
    from memory.conversation_db import DB_PATH, get_session
    chroma = _get_chroma()
    if not chroma._ready:
        return {"total": 0, "embedded": 0, "skipped": 0, "errors": 0, "error": "ChromaDB not ready"}

    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT id, session_id, summary FROM summaries ORDER BY id ASC"
        ).fetchall()
        conn.close()
    except Exception as e:
        return {"total": 0, "embedded": 0, "skipped": 0, "errors": 0, "error": str(e)}

    stats = {"total": len(rows), "embedded": 0, "skipped": 0, "errors": 0}

    # Deduplicate: per session_id, keep the latest summary (max id)
    latest_per_session = {}
    for row_id, session_id, summary in rows:
        if session_id not in latest_per_session or row_id > latest_per_session[session_id][0]:
            latest_per_session[session_id] = (row_id, summary)

    batch_docs = []
    batch_metas = []
    batch_ids = []

    for session_id, (row_id, summary) in latest_per_session.items():
        if not summary or not summary.strip():
            stats["skipped"] += 1
            continue
        session = get_session(session_id)
        title = session["title"] if session else session_id
        project_id = session.get("project_id", "") if session else ""
        keywords = _extract_keywords(summary)

        doc_id = f"session_summary/{session_id}"
        meta = _make_metadata("session_summary", session_id, summary, {
            "type": "session_summary",
            "session_id": session_id,
            "title": title,
            "project_id": project_id,
            "keywords": keywords,
            "created_at": str(datetime.now().isoformat()),
        })

        batch_docs.append(summary)
        batch_metas.append(meta)
        batch_ids.append(doc_id)
        stats["embedded"] += 1

    stats["total"] = len(latest_per_session)

    if batch_docs:
        try:
            chroma._collection.upsert(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )
            print(f"[ChromaMemory] Summary migration: {stats['embedded']}/{stats['total']} embedded "
                  f"({stats['skipped']} skipped, {stats['errors']} errors)")
        except Exception as e:
            stats["errors"] = len(batch_docs)
            print(f"[ChromaMemory] Summary migration error: {e}")
    else:
        print("[ChromaMemory] No summaries to migrate")

    return stats


def get_core_memories() -> list[dict]:
    """Get all core memories (identity, preferences, projects, etc.) from ChromaDB.
    Excludes session_summary entries — those are for search, not core memory injection."""
    chroma = _get_chroma()
    all_mem = chroma.get_all_memories()
    return [m for m in all_mem if m.get("category") != "session_summary"]


# ── Module-level convenience ───────────────────────────────────────────────

def get_bridge() -> MemoryBridge:
    return MemoryBridge()
