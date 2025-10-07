import pytest
from unittest.mock import Mock, patch
from query import ResearchAssistant

@pytest.fixture
def mock_indexer():
    indexer = Mock()
    indexer.search.return_value = [
        (0.1, "Relevant text 1", {"page": 1}),
        (0.2, "Relevant text 2", {"page": 2})
    ]
    return indexer

@pytest.fixture
def mock_summarizer():
    summarizer = Mock()
    summarizer.return_value = [{"summary_text": "Summarized answer"}]
    return summarizer

def test_answer_question(mock_indexer, mock_summarizer):
    with patch('query.pipeline', return_value=mock_summarizer):
        assistant = ResearchAssistant(mock_indexer)
        result = assistant.answer_question("Test question", top_k=2)
        assert "answer" in result
        assert "evidence" in result
        assert result["answer"] == "Summarized answer"
        assert len(result["evidence"]) == 2
        assert result["evidence"][0]["text"] == "Relevant text 1"
        assert result["evidence"][0]["page"] == 1
        assert result["evidence"][1]["text"] == "Relevant text 2"
        assert result["evidence"][1]["page"] == 2
        mock_indexer.search.assert_called_once_with("Test question", top_k=2)
        mock_summarizer.assert_called_once()
