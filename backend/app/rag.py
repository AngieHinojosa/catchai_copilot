import os
import numpy as np
from typing import List, Tuple
import chromadb
from google import genai

def get_db_path() -> str:
    if os.path.exists("/app"):
        base = "/app/data"
    else:
        here = os.path.abspath(os.path.dirname(__file__))
        base = os.path.abspath(os.path.join(here, "..", "..", "data"))
    os.makedirs(base, exist_ok=True)
    return base

def get_collection(name: str = "default"):
    client = chromadb.PersistentClient(path=os.path.join(get_db_path(), "chroma"))
    return client.get_or_create_collection(name=name)

def _gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en el entorno.")
    return genai.Client(api_key=api_key)

def embed_texts(texts: List[str]) -> List[List[float]]:
    c = _gemini_client()
    safe_texts = [(t if (t is not None and str(t).strip()) else " ") for t in texts]
    batch_size = 32
    out: List[List[float]] = []
    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i : i + batch_size]
        r = c.models.embed_content(model="text-embedding-004", contents=batch)
        out.extend([e.values for e in r.embeddings])
    return out

def ingest_docs(docs: List[Tuple[str, List[str], List[dict]]], collection_name: str = "default") -> List[str]:
    col = get_collection(collection_name)
    all_ids: List[str] = []
    for doc_id, chunks, metas in docs:
        if not chunks:
            continue
        if len(metas) != len(chunks):
            metas = (metas or [])
            metas = (metas + [{}] * len(chunks))[: len(chunks)]
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        vecs = embed_texts(chunks)
        col.add(ids=ids, documents=chunks, embeddings=vecs, metadatas=metas)
        all_ids.append(doc_id)
    return all_ids

def search(query: str, k: int = 5, collection_name: str = "default"):
    col = get_collection(collection_name)
    qvec = embed_texts([query])[0]
    res = col.query(
        query_embeddings=[qvec],
        n_results=max(1, int(k)),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    docs = res.get("documents", [[]])
    metas = res.get("metadatas", [[]])
    dists = res.get("distances", [[]])
    if docs and metas and dists:
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            out.append((doc, meta, float(dist)))
    return out

def answer_with_gemini(question: str, contexts: List[str]) -> str:
    c = _gemini_client()
    joined_ctx = "\n\n---\n\n".join(contexts)[:28000]
    prompt = (
        "Responde en español a la pregunta del usuario usando EXCLUSIVAMENTE el contexto proporcionado. "
        "Si no está en el contexto, di explícitamente que no aparece en los documentos.\n\n"
        f"Pregunta:\n{question}\n\n"
        f"Contexto:\n{joined_ctx}"
    )
    r = c.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return (getattr(r, "text", "") or "").strip()

def compare_with_gemini(a_title: str, a_text: str, b_title: str, b_text: str) -> str:
    c = _gemini_client()
    prompt = (
        f"Compara en español los documentos '{a_title}' y '{b_title}'. "
        "Sé claro y estructurado. Señala similitudes, diferencias y recomendaciones. "
        "No inventes información.\n\n"
        f"---\nDocumento A ({a_title}):\n{(a_text or '')[:8000]}\n\n"
        f"---\nDocumento B ({b_title}):\n{(b_text or '')[:8000]}"
    )
    r = c.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return (getattr(r, "text", "") or "").strip()

def cluster_embeddings(vectors: List[List[float]], n_clusters: int) -> List[int]:
    from sklearn.cluster import KMeans
    X = np.array(vectors, dtype=np.float32)
    if len(X) == 0:
        return []
    n_clusters = max(1, min(int(n_clusters), len(X)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels.tolist()

def label_cluster_with_gemini(examples: List[str]) -> str:
    c = _gemini_client()
    short_examples = [(x or "")[:1000] for x in examples[:10]]
    prompt = (
        "Dame un nombre corto (3–6 palabras), claro y en español para este conjunto de fragmentos. "
        "No inventes temas no presentes.\n\n"
        + "\n\n---\n\n".join(short_examples)
    )
    r = c.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return (getattr(r, "text", "Tema") or "Tema").strip()

def summarize_with_gemini(title: str, text: str) -> str:
    c = _gemini_client()
    prompt = (
        f"Resume en español, en 6–8 viñetas claras y fieles al contenido, el documento '{title}'. "
        "No inventes información.\n\n"
        f"Contenido:\n{(text or '')[:18000]}"
    )
    r = c.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return (getattr(r, "text", "") or "").strip()

def reset_collection(name: str = "default") -> None:
    client = chromadb.PersistentClient(path=os.path.join(get_db_path(), "chroma"))
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    client.get_or_create_collection(name=name)
