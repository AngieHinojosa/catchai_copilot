import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CatchAI Copilot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1100px; }
      .stButton>button { border-radius: 12px; padding: 0.5rem 1rem; }
      .chip { display:inline-block; padding:6px 10px; border:1px solid #2b2f3a; border-radius:999px; margin:4px 6px 0 0; font-size:12px;}
      .muted { color:#9aa3af; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = []
if "id_to_name" not in st.session_state:
    st.session_state.id_to_name = {}

st.sidebar.title("CatchAI Copilot")
st.sidebar.markdown("Copiloto conversacional sobre PDFs.")
st.sidebar.markdown(f"[Ver documentación]({API_URL}/docs)")
if st.sidebar.button("Limpiar sesión"):
    try:
        requests.post(f"{API_URL}/reset", timeout=60)
        st.toast("Índice del backend borrado")
    except Exception:
        st.sidebar.info("Sesión local limpiada. Si el backend no expone /reset, ignora este aviso.")
    st.session_state.doc_ids = []
    st.session_state.id_to_name = {}
    st.rerun()

st.title("CatchAI Copilot")

tabs = st.tabs(["Subir Documentos", "Chat", "Comparar", "Temas", "Resúmenes"])

with tabs[0]:
    st.subheader("Sube PDFs (máx. 5)")
    files = st.file_uploader("Arrastra y suelta o explora", type=["pdf"], accept_multiple_files=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Procesar"):
            if not files:
                st.warning("Sube al menos 1 PDF.")
            else:
                try:
                    up = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files[:5]]
                    with st.spinner("Procesando..."):
                        r = requests.post(f"{API_URL}/ingest", files=up, timeout=180)
                    r.raise_for_status()
                    data = r.json()
                    st.session_state.doc_ids = data.get("doc_ids", [])
                    try:
                        sres = requests.get(f"{API_URL}/summaries", timeout=120).json()
                        for s in sres.get("summaries", []):
                            st.session_state.id_to_name[s["doc_id"]] = s["filename"]
                    except Exception:
                        pass
                    st.toast("¡Documentos indexados!")
                    st.success(f"¡Listo! Indexados: {', '.join(st.session_state.doc_ids)}")
                except requests.exceptions.RequestException as e:
                    st.error(f"No pude conectar al API: {e}")
    with col2:
        if st.session_state.doc_ids:
            names = [st.session_state.id_to_name.get(i, i) for i in st.session_state.doc_ids]
            st.markdown("**Documentos en sesión:**")
            for i, n in zip(st.session_state.doc_ids, names):
                st.markdown(f"<span class='chip'>{n} — {i}</span>", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Pregunta")
    q = st.text_input("Escribe tu pregunta aquí")
    if st.button("Preguntar"):
        if not q.strip():
            st.warning("Escribe una pregunta.")
        else:
            try:
                payload = {"question": q, "k": 5}
                with st.spinner("Buscando y generando respuesta…"):
                    r = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                st.markdown("#### Respuesta")
                st.markdown(data.get("answer", ""))
                srcs = data.get("sources", [])
                if srcs:
                    st.markdown("#### Fuentes")
                    df = pd.DataFrame(srcs)
                    df = df.rename(columns={"filename": "archivo", "doc_id": "doc_id", "chunk": "fragmento", "score": "relevancia"})
                    df["archivo"] = df["archivo"].fillna("")
                    df["relevancia"] = df["relevancia"].map(lambda x: round(float(x), 3))
                    st.dataframe(df[["archivo", "doc_id", "fragmento", "relevancia"]], use_container_width=True, hide_index=True)
            except requests.exceptions.RequestException as e:
                st.error(f"No pude conectar al API: {e}")
            except ValueError:
                st.error("La respuesta del API no es JSON válido.")

with tabs[2]:
    st.subheader("Comparar dos documentos")
    if not st.session_state.doc_ids:
        st.info("Aún no hay documentos indexados. Sube algunos en la pestaña **Ingesta**.")
    else:
        options = [f"{st.session_state.id_to_name.get(i, 'Documento')} ({i})" for i in st.session_state.doc_ids]
        sel = st.multiselect("Elige exactamente 2", options, max_selections=2)
        if st.button("Comparar", disabled=len(sel) != 2):
            ids = [s.split("(")[-1].rstrip(")") for s in sel]
            try:
                with st.spinner("Comparando…"):
                    r = requests.post(f"{API_URL}/compare", json={"doc_ids": ids}, timeout=180)
                r.raise_for_status()
                st.markdown(r.json().get("comparison", ""))
            except requests.exceptions.RequestException as e:
                st.error(f"No pude conectar al API: {e}")
            except ValueError:
                st.error("La respuesta del API no es JSON válido.")

with tabs[3]:
    st.subheader("Agrupar en temas")
    c = st.number_input("Número de temas a formar", 2, 10, 3, step=1,
                        help="Cantidad de grupos (clusters) que KMeans intentará encontrar sobre los fragmentos.")
    s = st.number_input("Ejemplos por tema", 1, 10, 3, step=1,
                        help="Cuántos fragmentos de muestra se listan por tema para entenderlo.")
    if st.button("Ver temas"):
        try:
            with st.spinner("Calculando…"):
                r = requests.get(
                    f"{API_URL}/topics",
                    params={"n_clusters": int(c), "samples_per_cluster": int(s)},
                    timeout=240
                )
            r.raise_for_status()
            data = r.json()
            labs = data.get("samples", [])
            if not labs:
                st.info("No hay datos suficientes.")
            else:
                for grp in labs:
                    with st.container(border=True):
                        st.markdown(f"**Tema {grp['label']} — {grp.get('name','Tema')}**")
                        for ex in grp.get("examples", [])[:s]:
                            st.markdown(f"<div class='muted'>{ex[:300]}…</div>", unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"No pude conectar al API: {e}")
        except ValueError:
            st.error("La respuesta del API no es JSON válido.")

with tabs[4]:
    st.subheader("Resúmenes por documento")
    if st.button("Generar / refrescar"):
        try:
            with st.spinner("Resumiendo…"):
                r = requests.get(f"{API_URL}/summaries", timeout=240)
            r.raise_for_status()
            data = r.json()
            lst = data.get("summaries", [])
            if not lst:
                st.info("No hay documentos.")
            for item in lst:
                st.session_state.id_to_name[item["doc_id"]] = item["filename"]
                with st.expander(f"{item['filename']} — {item['doc_id']}"):
                    st.markdown(item["summary"])
        except requests.exceptions.RequestException as e:
            st.error(f"No pude conectar al API: {e}")
        except ValueError:
            st.error("La respuesta del API no es JSON válido.")
