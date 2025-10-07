from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from config import EMBEDDING_MODEL

class SemanticIndexer:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the semantic indexer with a sentence transformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.metadata = []

    def build_index(self, texts: list[str], metadatas: list[dict]):
        """
        Build the FAISS index from texts and their metadata.

        Args:
            texts (list[str]): List of text chunks to embed.
            metadatas (list[dict]): Corresponding metadata for each text chunk.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.texts = texts
        self.metadata = metadatas

    def add_texts(self, texts: list[str], metadatas: list[dict]):
        """
        Add new texts to the existing index.

        Args:
            texts (list[str]): List of new text chunks to embed and add.
            metadatas (list[dict]): Corresponding metadata for each new text chunk.
        """
        if self.index is None:
            raise ValueError("Index not built. Use build_index first.")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    def search(self, query: str, top_k: int = 5):
        """
        Search the index for the most similar text chunks to the query.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to return.

        Returns:
            list of tuples: (score, text, metadata)
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append((dist, self.texts[idx], self.metadata[idx]))
        return results

    def save(self, path: str):
        """
        Save the indexer to a file.

        Args:
            path (str): Path to save the indexer.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Load the indexer from a file.

        Args:
            path (str): Path to load the indexer from.

        Returns:
            SemanticIndexer: The loaded indexer.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
