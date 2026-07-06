import traceback
try:
    from ragas.metrics import faithfulness
    print("Imported faithfulness")
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from ragas import evaluate

    _local_llm = ChatOllama(model="llama3.2", temperature=0)
    _evaluator_llm = LangchainLLMWrapper(_local_llm)
    _evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )

    data = {
        "question": ["What is a Horcrux?"],
        "contexts": [["A Horcrux is an object in which a Dark wizard has hidden a fragment of his soul in order to become immortal."]],
        "answer": ["A Horcrux is an object to hide soul."],
        "ground_truth": ["A Horcrux is an object in which a Dark wizard has hidden a fragment of his soul in order to become immortal."]
    }
    dataset = Dataset.from_dict(data)

    metrics = [faithfulness]
    for metric in metrics:
        metric.llm = _evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = _evaluator_embeddings
            
    print("Evaluating...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_evaluator_llm,
        embeddings=_evaluator_embeddings
    )
    print(result)
except Exception as e:
    traceback.print_exc()
