import streamlit as st
import os
import threading

INDEX_DIR = "faiss_index"
RAG_MODEL = "qwen2.5:0.5b"

# ─── Page config FIRST — renders immediately ───────────────────
st.set_page_config(page_title="Harry Potter RAG", page_icon="⚡", layout="wide")

# ─── CSS Styling ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a0a2e, #2d1b4e);
        border: 1px solid #7b2d8b;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-label {
        color: #c792ea;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #ffd700;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .eval-header {
        color: #c792ea;
        font-size: 0.85rem;
        margin-bottom: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Title renders immediately ──────────────────────────────────
st.title("⚡ Harry Potter Lore Assistant")
st.write("Ask me anything about Harry Potter movies, lore, or behind-the-scenes production notes!")

# ─── Initialize session state ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = []
if "app_ready" not in st.session_state:
    st.session_state.app_ready = False

# ─── Sidebar (always visible) ───────────────────────────────────
with st.sidebar:
    st.header("📊 Eval Scores (Per Query)")
    if st.session_state.eval_scores:
        for i, score in enumerate(reversed(st.session_state.eval_scores[-5:])):
            label = score['question'][:40] + "..." if len(score['question']) > 40 else score['question']
            st.markdown(f"**Q{len(st.session_state.eval_scores) - i}:** _{label}_")
            col1, col2 = st.columns(2)
            col1.metric("Faithfulness", f"{score.get('faithfulness', 0):.2f}")
            col2.metric("Relevancy", f"{score.get('answer_relevancy', 0):.2f}")
            col1.metric("Precision", f"{score.get('context_precision', 0):.2f}")
            col2.metric("Recall", f"{score.get('context_recall', 0):.2f}")
            st.divider()
    else:
        st.info("Ask a question to see RAGAS evaluation scores here.")

# ─── STEP 1: Lazy DB init (cached, runs once) ──────────────────
@st.cache_resource(show_spinner=False)
def init_database():
    try:
        import database
        database.init_db()
        return True
    except Exception as e:
        return False

# ─── STEP 2: Lazy vectorstore (cached, runs once) ──────────────
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Loads FAISS index with Ollama embeddings. No cold start — Ollama is always running."""
    if not os.path.exists(INDEX_DIR):
        return None
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# ─── STEP 3: Build RAG chain (lazy, uses cached vectorstore) ────
@st.cache_resource(show_spinner=False)
def get_rag_chain(_vectorstore):
    """Builds the qwen2.5:0.5b RAG chain. Light — Ollama is called per-query."""
    from langchain_ollama import ChatOllama
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOllama(model=RAG_MODEL, temperature=0)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    system_prompt = (
        "You are a Harry Potter expert. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, just say that you don't know. "
        "Try to cite the sources of your information from the context if possible.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# ─── Background evaluation (llama3.2 loaded only here) ──────────
def run_evaluation_async(question, answer, contexts, db_connected):
    """Runs RAGAS evaluation in the background using llama3.2 as judge."""
    try:
        import evaluator
        ground_truth = evaluator.generate_ground_truth(question, contexts)
        metrics = evaluator.evaluate_rag_run(question, answer, contexts, ground_truth)
        if metrics:
            print(f"[EVAL COMPLETE] {metrics}")
            
            # Save to Streamlit session state
            import streamlit as st
            score = metrics.copy()
            score["question"] = question
            st.session_state.eval_scores.append(score)

            if db_connected:
                import database
                database.log_evaluation(
                    model_name=RAG_MODEL,
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    metrics=metrics
                )
    except Exception as e:
        print(f"[EVAL ERROR] {e}")

# ─── Load Chat History (always visible) ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Progressive loading ─────────────────────────────────────────
# Only load vectorstore when app starts — show spinner in main area
if not os.path.exists(INDEX_DIR):
    st.error(f"⚠️ Index directory '{INDEX_DIR}' not found. Please run `index_data.py` first.")
    st.stop()

with st.spinner("🔮 Loading knowledge base (first-time setup, takes ~30s)..."):
    vectorstore = load_vectorstore()

if vectorstore is None:
    st.error("Failed to load vectorstore.")
    st.stop()

# DB init (fast — just a connection check)
db_connected = init_database()
if not db_connected:
    st.sidebar.warning("⚠️ PostgreSQL not connected. Scores logged in session only.")

# RAG chain (very fast — just wires objects together)
rag_chain = get_rag_chain(vectorstore)

# ─── Chat Input ──────────────────────────────────────────────────
if user_input := st.chat_input("What is a Horcrux?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Restricted Section..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                retrieved_docs = response["context"]
                contexts = [doc.page_content for doc in retrieved_docs]

                st.markdown(answer)

                # Show Sources
                with st.expander("📚 Retrieved Sources"):
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get("source", "Unknown")
                        st.write(f"**Source {i+1}:** `{source}`")
                        st.caption(doc.page_content)

                # Kick off llama3.2 evaluation in background thread (loads lazily)
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                eval_thread = threading.Thread(
                    target=run_evaluation_async,
                    args=(user_input, answer, contexts, db_connected),
                    daemon=True
                )
                add_script_run_ctx(eval_thread)
                eval_thread.start()
                st.info("🔍 RAGAS evaluation running in background via `llama3.2`... Check sidebar after next query.")

            except Exception as e:
                answer = f"Error: {str(e)}. Make sure Ollama is running and 'qwen2.5:0.5b' is downloaded."
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
