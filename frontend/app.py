import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Semantic Search", layout="centered")
st.title("Semantic Search Engine")
st.caption("pgvector + cross-encoder reranking — no LangChain")

tab1, tab2 = st.tabs(["Search", "Upload Documents"])

# --- SEARCH TAB ---
with tab1:
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        st.info(f"{stats['total_documents']} documents in the index")
    except:
        st.warning("API not reachable. Is it running?")

    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of results", 1, 10, 5)

    if query:
        with st.spinner("Searching..."):
            res = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k})
            if res.status_code == 200:
                data = res.json()
                meta = data.get("meta", {})
                st.caption(
                    f"⏱ Retrieval: {meta.get('retrieval_ms')}ms | "
                    f"Reranking: {meta.get('rerank_ms')}ms | "
                    f"Total: {meta.get('total_ms')}ms"
                )
                for i, r in enumerate(data["results"]):
                    st.markdown(f"**{i+1}.** {r['content']}")
                    st.caption(f"Score: {r['score']:.4f}")
                    st.divider()
            else:
                st.error("Search failed.")

# --- UPLOAD TAB ---
with tab2:
    st.subheader("Upload a document to index")
    st.caption("Supported formats: .txt, .pdf, .csv")

    uploaded_files = st.file_uploader(
        "Choose files", type=["txt", "pdf", "csv"], accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            st.write(f"**{f.name}** — {round(f.size / 1024, 1)} KB")

        if st.button("Ingest all into search index"):
            with st.spinner("Parsing and embedding..."):
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type))
                    for f in uploaded_files
                ]
                res = requests.post(f"{API_URL}/ingest", files=files_payload)
                if res.status_code == 200:
                    data = res.json()
                    for result in data["files"]:
                        if "error" in result:
                            st.error(f"{result['source']}: {result['error']}")
                        else:
                            st.success(f"{result['source']}: {result['ingested']} chunks ingested")
                else:
                    st.error("Ingestion failed.")