import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os



def test_query_command():
    with patch('main.SemanticIndexer.load') as mock_load, \
         patch('main.ResearchAssistant') as mock_assistant_class:
        mock_indexer = Mock()
        mock_load.return_value = mock_indexer
        mock_assistant = Mock()
        mock_assistant.answer_question.return_value = {
            "answer": "Test answer",
            "evidence": [{"text": "Evidence", "page": 1}]
        }
        mock_assistant_class.return_value = mock_assistant

        with patch('sys.argv', ['main.py', 'query', 'What is AI?', '--index-path', 'test.pkl', '--top-k', '3']):
            from main import main
            main()

        mock_load.assert_called_once_with('test.pkl')
        mock_assistant.answer_question.assert_called_once_with('What is AI?', top_k=3)

def test_query_missing_index():
    with patch('main.SemanticIndexer.load', side_effect=FileNotFoundError):
        with patch('sys.argv', ['main.py', 'query', 'Question']):
            with pytest.raises(SystemExit):
                from main import main
                main()
