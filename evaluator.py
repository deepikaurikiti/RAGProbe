"""
Lazy-loading RAGAS evaluator.
Models are NOT initialized at import time — they are loaded only when
evaluation is actually triggered, preventing slow startup.
"""

_local_llm = None
_evaluator_llm = None
_evaluator_embeddings = None

def _get_evaluator_components():
    """Lazy-loads the evaluator LLM and embeddings only on first use."""
    global _local_llm, _evaluator_llm, _evaluator_embeddings

    if _local_llm is None:
        from langchain_community.chat_models import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        print("[EVALUATOR] Loading llama3.2 as evaluation judge...")
        _local_llm = ChatOllama(model="llama3.2", temperature=0)
        _evaluator_llm = LangchainLLMWrapper(_local_llm)
        # Reuse the same embedding model already cached on disk
        _evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        print("[EVALUATOR] Evaluator components loaded.")

    return _local_llm, _evaluator_llm, _evaluator_embeddings


def generate_ground_truth(question, contexts):
    """
    Generates a ground truth answer using llama3.2 based on retrieved contexts.
    """
    local_llm, _, _ = _get_evaluator_components()
    context_str = "\n\n".join(contexts)
    prompt = (
        f"You are a Harry Potter expert examiner. Based strictly on the following retrieved context, "
        f"write the most accurate, concise, and ideal ground-truth answer to the question.\n"
        f"Do not invent anything not in the context.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Ideal Ground-Truth Answer:"
    )
    try:
        response = local_llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[EVALUATOR] Error generating ground truth: {e}")
        return "Could not generate ground truth."


def evaluate_rag_run(question, answer, contexts, ground_truth):
    """
    Runs RAGAS evaluation metrics on a single query run.
    Returns a dict with metric scores, or None if evaluation fails.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

    _, evaluator_llm, evaluator_embeddings = _get_evaluator_components()

    data = {
        "question": [question],
        "contexts": [contexts],
        "answer": [answer],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for metric in metrics:
        metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        def safe_float(val):
            """Extract a scalar float whether val is a float, int, list, or None."""
            if val is None:
                return 0.0
            if isinstance(val, (list, tuple)):
                val = val[0] if len(val) > 0 else 0.0
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        return {
            "faithfulness": safe_float(result["faithfulness"]),
            "answer_relevancy": safe_float(result["answer_relevancy"]),
            "context_recall": safe_float(result["context_recall"]),
            "context_precision": safe_float(result["context_precision"]),
        }
    except Exception as e:
        print(f"[EVALUATOR] RAGAS evaluation failed: {e}")
        return None
