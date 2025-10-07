import os
import tempfile
import pytest
from indexer import SemanticIndexer

@pytest.fixture
def sample_texts():
    return ["This is a test.", "Another test sentence.", "More text data."]

@pytest.fixture
def sample_metadatas():
    return [{"page": 1}, {"page": 2}, {"page": 3}]

def test_build_and_search(sample_texts, sample_metadatas):
    indexer = SemanticIndexer()
    indexer.build_index(sample_texts, sample_metadatas)
    results = indexer.search("test", top_k=2)
    assert len(results) == 2
    for dist, text, metadata in results:
        assert text in sample_texts
        assert metadata in sample_metadatas

def test_save_and_load(sample_texts, sample_metadatas):
    indexer = SemanticIndexer()
    indexer.build_index(sample_texts, sample_metadatas)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    indexer.save(path)
    loaded_indexer = SemanticIndexer.load(path)
    os.remove(path)
    assert loaded_indexer.texts == sample_texts
    assert loaded_indexer.metadata == sample_metadatas
    results = loaded_indexer.search("test", top_k=2)
    assert len(results) == 2
