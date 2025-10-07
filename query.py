from transformers import pipeline
from indexer import SemanticIndexer
from config import SUMMARIZATION_MODEL


class ResearchAssistant:
    def __init__(self, indexer: SemanticIndexer, model_name: str = SUMMARIZATION_MODEL):
        """
        Initialize the research assistant with a semantic indexer and a summarization model.
        """
        self.indexer = indexer
        self.summarizer = pipeline("summarization", model=model_name)

    def answer_question(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a question by retrieving relevant chunks and generating a concise, evidence-backed answer.

        Args:
            question (str): The question to answer.
            top_k (int): Number of top relevant chunks to retrieve.

        Returns:
            dict: {
                "answer": str,
                "evidence": list of dicts with "text" and "page"
            }
        """
        results = self.indexer.search(question, top_k=top_k)
        combined_text = " ".join([res[1] for res in results])
        summary = self.summarizer(combined_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

        evidence = []
        for _, text, metadata in results:
            evidence.append({"text": text, "page": metadata.get("page", None)})
        return {"answer": summary, "evidence": evidence}
