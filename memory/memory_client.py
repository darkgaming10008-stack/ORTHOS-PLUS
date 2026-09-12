import os
import numpy as np

SERVICE_URL = "http://127.0.0.1:9876"
_healthy: bool | None = None
_IS_MEMORY_SERVICE = os.environ.get("MEMORY_SERVICE_PROCESS") == "1"


def health() -> dict | None:
    try:
        import requests
        resp = requests.get(f"{SERVICE_URL}/health", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _is_healthy() -> bool:
    if _IS_MEMORY_SERVICE:
        return False
    global _healthy
    if _healthy is None:
        _healthy = health() is not None
    return _healthy


def invalidate_cache():
    global _healthy
    _healthy = None


def memory_embed(texts: list[str]) -> np.ndarray:
    import requests
    resp = requests.post(
        f"{SERVICE_URL}/embed",
        json={"texts": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["vectors"], dtype=np.float32)


def memory_ner(text: str) -> list[str]:
    import requests
    resp = requests.post(
        f"{SERVICE_URL}/ner",
        json={"text": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("entities", [])


def memory_ner_typed(text: str) -> list[dict]:
    """Ask the memory service for typed entities: [{name, type}]."""
    import requests
    resp = requests.post(
        f"{SERVICE_URL}/ner_typed",
        json={"text": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("entities", [])


def memory_search_vector(query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
    import requests
    resp = requests.post(
        f"{SERVICE_URL}/search_vector",
        json={"query": query, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        (r["text"], r["source"], r["score"])
        for r in data.get("results", [])
    ]


def memory_search_bm25(query: str, top_k: int = 10) -> list[tuple[str, dict, float]]:
    import requests
    resp = requests.post(
        f"{SERVICE_URL}/search_bm25",
        json={"query": query, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        (r["text"], r["source"], r["score"])
        for r in data.get("results", [])
    ]


def memory_rerank(query: str, texts: list[str], top_k: int | None = None) -> list[int]:
    """Ask the memory service to cross-encoder rerank candidate texts.

    Returns the candidate indices in re-ranked order (service-side model).
    """
    import requests
    resp = requests.post(
        f"{SERVICE_URL}/rerank",
        json={"query": query, "texts": texts, "top_k": top_k},
    )
    resp.raise_for_status()
    data = resp.json()
    order = data.get("order")
    if order is None:
        raise RuntimeError("memory service /rerank returned no order")
    return order
