# streamlit_app.py
# RAG Document Analyzer by Shivendra — Personalized, Enhanced, Streamlit App
# Compatible with OpenAI Python SDK >=1.0

import streamlit as st
from pathlib import Path
import tempfile
import base64
from qa_agent import generate_summary, generate_insights, generate_mcq, answer_question, build_retrieval_index

# -------------------------- Page Config --------------------------
st.set_page_config(
    page_title="📊 RAG Document Analyzer — Shivendra",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- Helper Utilities --------------------------
def save_uploaded_file(uploaded_file, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.read())
    return dst_path

def make_download_link(text: str, filename: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    return f"data:file/txt;base64,{b64}"

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    valid_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    model = st.selectbox("Choose OpenAI Model", valid_models, index=0)

    temp = st.slider("Response Temperature", 0.0, 1.0, 0.3, 0.05)
    chunk_size = st.number_input("Document Chunk Size (tokens)", 256, 4096, 1024, 64)
    reindex = st.button("🔄 Rebuild Document Index")

    st.markdown("---")
    st.markdown("**📌 Tips for Use:**")
    st.markdown("- Upload your PDF or CSV file.")
    st.markdown("- Rebuild the index after uploading.")
    st.markdown("- Use tabs to summarize, chat, or generate MCQs.")

# -------------------------- Header --------------------------
col1, col2 = st.columns([8, 2])
with col1:
    st.title("📄 Smart Document Analyzer")
    st.caption("By Shivendra — Summarize, Extract Insights, Chat & Generate MCQs from your financial docs.")
with col2:
    st.image("https://static.streamlit.io/images/brand/streamlit-mark-color.png", width=72)

st.markdown("---")

# -------------------------- File Upload --------------------------
uploaded_file = st.file_uploader("📤 Upload Your Document (PDF/CSV)", type=["pdf", "csv"], accept_multiple_files=False)

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

if uploaded_file:
    tmp = Path(tempfile.gettempdir()) / "uploaded_document"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / uploaded_file.name
    save_uploaded_file(uploaded_file, dst)
    st.session_state.uploaded_path = str(dst)
    st.success(f"✅ File uploaded: `{uploaded_file.name}`")

# -------------------------- Tabs --------------------------
tabs = st.tabs(["📘 Preview", "🧠 Analyze", "💬 Chat", "🎓 MCQs & Export"])

# -------------------------- Document Tab --------------------------
with tabs[0]:
    st.subheader("📄 Document Info & Preview")
    if st.session_state.uploaded_path:
        file_name = Path(st.session_state.uploaded_path).name
        size_kb = Path(st.session_state.uploaded_path).stat().st_size // 1024
        st.write(f"**File Name:** `{file_name}` ({size_kb} KB)")
        st.write(f"**Model:** `{model}` | **Chunk Size:** `{chunk_size}`")

        with st.expander("📝 Show Raw Text Preview (First 8000 Characters)"):
            try:
                with open(st.session_state.uploaded_path, "rb") as f:
                    _b = f.read()
                if file_name.endswith(".pdf"):
                    st.info("PDF text will be extracted during processing.")
                else:
                    st.text(_b[:8000].decode(errors="replace"))
            except Exception:
                st.error("❌ Unable to preview this file type.")
    else:
        st.info("Please upload a document to begin.")

# -------------------------- Actions Tab --------------------------
with tabs[1]:
    st.subheader("🧠 Analyze Document")
    if not st.session_state.uploaded_path:
        st.warning("⚠️ Please upload a document first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📄 Create Summary"):
                with st.spinner("Summarizing..."):
                    summary = generate_summary(st.session_state.uploaded_path, model=model, temperature=temp)
                    st.session_state.last_summary = summary
                    st.success("✅ Summary Generated")
                    st.write(summary)
        with c2:
            if st.button("💡 Extract Insights"):
                with st.spinner("Extracting insights..."):
                    insights = generate_insights(st.session_state.uploaded_path, model=model, chunk_size=chunk_size)
                    st.session_state.last_insights = insights
                    st.success("✅ Insights Ready")
                    st.write(insights)
        with c3:
            if st.button("🎓 Create MCQs"):
                with st.spinner("Generating MCQs..."):
                    mcqs = generate_mcq(st.session_state.uploaded_path, num_questions=10, model=model)
                    st.session_state.last_mcqs = mcqs.split("\n\n")
                    st.success("✅ MCQs Ready")
                    for i, q in enumerate(st.session_state.last_mcqs):
                        st.markdown(f"**Q{i+1}:** {q}")

        if reindex:
            with st.spinner("Rebuilding document index..."):
                result = build_retrieval_index(st.session_state.uploaded_path, chunk_size=chunk_size)
                st.success(result)

# -------------------------- Chat Tab --------------------------
with tabs[2]:
    st.subheader("💬 Chat with Document")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.uploaded_path:
        st.info("Upload a document to start chatting.")
    else:
        query = st.text_input("Ask something about the document:")
        if st.button("💬 Ask"):
            if query:
                with st.spinner("Thinking..."):
                    answer = answer_question(st.session_state.uploaded_path, query, model=model)
                    st.session_state.chat_history.append({"q": query, "a": answer})
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []

        for turn in reversed(st.session_state.chat_history[-10:]):
            st.markdown(f"**Q:** {turn['q']}")
            st.markdown(f"**A:** {turn['a']}")

# -------------------------- MCQs & Export Tab --------------------------
with tabs[3]:
    st.subheader("🎓 Review & Export Results")

    # Editable MCQ list
    if "last_mcqs" in st.session_state and st.session_state.last_mcqs:
        st.markdown("### MCQs:")
        edited_mcqs = []
        for idx, q in enumerate(st.session_state.last_mcqs):
            with st.expander(f"MCQ {idx+1}"):
                edited = st.text_area("Edit Question:", value=q, key=f"mcq_{idx}")
                col1, col2 = st.columns([5, 1])
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        continue  # Skip adding this one
                edited_mcqs.append(edited)
        st.session_state.last_mcqs = edited_mcqs

        st.download_button("⬇️ Download MCQs", "\n\n".join(edited_mcqs), file_name="mcqs_shivendra.txt")
    else:
        st.info("No MCQs generated yet.")

    if st.session_state.get("last_summary"):
        st.download_button("⬇️ Download Summary", st.session_state.last_summary, file_name="summary_shivendra.txt")

    if st.session_state.get("last_insights"):
        st.download_button("⬇️ Download Insights", st.session_state.last_insights, file_name="insights_shivendra.txt")

st.markdown("---")
st.caption("🚀 Built by Shivendra | Future Upgrades: LangChain, Async Processing, and Custom Retrieval UI")
