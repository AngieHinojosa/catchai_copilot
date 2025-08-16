import os
import io
import uuid
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import IngestResponse, ChatRequest, ChatAnswer, CompareRequest, TopicsResponse, SummariesResponse
from .chunking import pdf_to_text, chunk_text
from .rag import ingest_docs, search, answer_with_gemini, compare_with_gemini, cluster_embeddings, label_cluster_with_gemini, summarize_with_gemini, get_collection, embed_texts, reset_collection

app = FastAPI(title="CatchAI Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No se recibieron archivos")
        if len(files) > 5:
            files = files[:5]

        docs = []
        for f in files:
            content = await f.read()
            path = os.path.join(tempfile.gettempdir(), f.filename)
            with open(path, "wb") as out:
                out.write(content)
            text = pdf_to_text(path)
            chunks = chunk_text(text)
            metas = [{"filename": f.filename, "doc_id": "", "chunk": i} for i in range(len(chunks))]
            doc_id = str(uuid.uuid4())[:8]
            for m in metas:
                m["doc_id"] = doc_id
            docs.append((doc_id, chunks, metas))

        ids = ingest_docs(docs)
        return IngestResponse(doc_ids=ids)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en /ingest: {e}")

@app.post("/chat", response_model=ChatAnswer)
async def chat(req: ChatRequest):
    try:
        hits = search(req.question, k=req.k)
        contexts = []
        sources = []
        for doc, meta, dist in hits:
            tag = f"[doc:{meta.get('doc_id')}|{meta.get('filename')}|chunk:{meta.get('chunk')}]"
            contexts.append(tag + "\n" + doc)
            sources.append({
                "doc_id": meta.get("doc_id"),
                "filename": meta.get("filename"),
                "chunk": meta.get("chunk"),
                "score": float(dist)
            })
        ans = answer_with_gemini(req.question, contexts)
        return ChatAnswer(answer=ans, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en /chat: {e}")

@app.post("/compare")
async def compare(req: CompareRequest):
    try:
        col = get_collection()
        items = col.get(include=["documents", "metadatas"])
        by_doc = {}
        for doc, meta in zip(items["documents"], items["metadatas"]):
            did = meta.get("doc_id")
            if did in req.doc_ids:
                by_doc.setdefault(did, {"title": meta.get("filename"), "text": []})
                by_doc[did]["text"].append(doc)
        keys = list(by_doc.keys())[:2]
        if len(keys) < 2:
            return {"comparison": "Se requieren 2 doc_ids"}
        a, b = keys
        at = by_doc[a]["title"]
        bt = by_doc[b]["title"]
        ax = "\n".join(by_doc[a]["text"])
        bx = "\n".join(by_doc[b]["text"])
        txt = compare_with_gemini(at, ax, bt, bx)
        return {"comparison": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en /compare: {e}")

@app.get("/topics", response_model=TopicsResponse)
async def topics(n_clusters: int = 3, samples_per_cluster: int = 3):
    col = get_collection()
    items = col.get(include=["documents", "embeddings", "metadatas"])
    vecs = items["embeddings"]
    labels = cluster_embeddings(vecs, n_clusters)
    groups = {}
    for doc, meta, label in zip(items["documents"], items["metadatas"], labels):
        groups.setdefault(label, []).append((doc, meta))
    samples = []
    for label, docs in groups.items():
        ex = [d[0] for d in docs[:samples_per_cluster]]
        name = label_cluster_with_gemini(ex)
        samples.append({"label": label, "name": name.strip(), "examples": ex})
    return TopicsResponse(n_clusters=n_clusters, labels=labels, samples=samples)

@app.get("/summaries", response_model=SummariesResponse)
async def summaries():
    col = get_collection()
    items = col.get(include=["documents", "metadatas"])
    merged = {}
    for doc, meta in zip(items["documents"], items["metadatas"]):
        did = meta.get("doc_id")
        merged.setdefault(did, {"title": meta.get("filename"), "text": []})
        merged[did]["text"].append(doc)
    out = []
    for did, v in merged.items():
        txt = "\n".join(v["text"])[:20000]
        s = summarize_with_gemini(v["title"], txt)
        out.append({"doc_id": did, "filename": v["title"], "summary": s})
    return SummariesResponse(summaries=out)

@app.post("/reset")
def reset_index():
    reset_collection("default")
    return {"status": "ok"}