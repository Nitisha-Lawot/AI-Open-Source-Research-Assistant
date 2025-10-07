# AI Open Source Research Assistant

An AI-powered research assistant that ingests PDF documents, builds a semantic index, and provides concise, evidence-backed answers to questions with supporting passages and page references.

## Features

- **PDF Ingestion**: Extracts text from PDF files while preserving page information.
- **Semantic Indexing**: Uses sentence transformers and FAISS for efficient semantic search.
- **Question Answering**: Retrieves relevant passages and generates answers using Hugging Face models.
- **Evidence-Based Responses**: Provides answers with exact supporting passages and page references.
- **CLI Interface**: Easy-to-use command-line interface for ingestion and querying.
- **Logging**: Comprehensive logging for observability and debugging.
- **Error Handling**: Robust error handling and input validation.
- **Unit Tests**: Comprehensive test suite for reliability.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-open-source-research-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install python-dotenv for environment variable support:
   ```bash
   pip install python-dotenv
   ```

4. (Optional) Install pytest for running tests:
   ```bash
   pip install pytest
   ```

## Usage

### Web Interface (Recommended)

For an easy-to-use web interface, run the Streamlit app:

```bash
streamlit run app.py
```

This will start a local web server. Open the provided URL (usually http://localhost:8501) in your browser. Upload a PDF, and ask questions about its content directly in the interface.

### Command Line Interface

#### Ingest PDFs and Build Index

To process PDF files and build the semantic index:

```bash
python main.py ingest path/to/document1.pdf path/to/document2.pdf --index-path my_index.pkl
```

This will extract text, split into chunks, embed them, and save the index to `my_index.pkl`.

Example output:
```
2023-10-01 12:00:00,000 - INFO - Starting ingestion of 2 PDF files
2023-10-01 12:00:05,000 - INFO - Processing PDF: path/to/document1.pdf
2023-10-01 12:00:10,000 - INFO - Extracted 50 chunks from path/to/document1.pdf
2023-10-01 12:00:15,000 - INFO - Building index with 100 text chunks
2023-10-01 12:00:20,000 - INFO - Index built and saved to my_index.pkl
Index built and saved to my_index.pkl
```

#### Query the Assistant

To ask a question and get an answer:

```bash
python main.py query "What is the main topic of the document?" --index-path my_index.pkl --top-k 5
```

This will output the answer and list the supporting evidence with page numbers.

Example output:
```
Answer: The main topic of the document is artificial intelligence and its applications in research.

Evidence:
1. Page 1: Artificial intelligence (AI) is a field of computer science that aims to create machines capable of intelligent behavior...
2. Page 3: AI has numerous applications in research, including data analysis, pattern recognition, and automated reasoning...
```

#### Performance Benchmarks

To run performance benchmarks:

```bash
python main.py benchmark --index-path my_index.pkl --num-queries 10
```

This measures the time to load the index and run queries.

### Running Tests

To run the unit tests:

```bash
python -m pytest tests/
```

Or for a specific test file:

```bash
python -m pytest tests/test_indexer.py
```

## Configuration

The assistant supports configuration via environment variables. Create a `.env` file in the project root to override defaults:

```env
# Model configurations
EMBEDDING_MODEL=all-MiniLM-L6-v2
SUMMARIZATION_MODEL=facebook/bart-large-cnn

# Processing parameters
CHUNK_SIZE=500

# File paths
DEFAULT_INDEX_PATH=index.pkl

# Logging
LOG_LEVEL=INFO
```

Available options:
- **EMBEDDING_MODEL**: Sentence transformer model for embeddings (default: 'all-MiniLM-L6-v2')
- **SUMMARIZATION_MODEL**: Hugging Face model for summarization (default: 'facebook/bart-large-cnn')
- **CHUNK_SIZE**: Text chunk size in characters (default: 500)
- **DEFAULT_INDEX_PATH**: Default path for index file (default: 'index.pkl')
- **LOG_LEVEL**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO)

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**:
   - Ensure PDFs are not password-protected.
   - Check if PDFs contain selectable text (not just images).
   - Solution: Use OCR tools if PDFs are image-based.

2. **Index File Not Found**:
   - Run the `ingest` command before querying.
   - Check the `--index-path` argument.

3. **Out of Memory Errors**:
   - Reduce chunk size or process fewer PDFs at once.
   - Use a smaller embedding model.

4. **Slow Performance**:
   - Use GPU if available (install torch with CUDA support).
   - Reduce top-k for queries.

### Logs

The application logs to console with timestamps. For more detailed logging, set the environment variable:

```bash
export LOG_LEVEL=DEBUG
```

## Dependencies

- pdfplumber: For PDF text extraction
- sentence-transformers: For generating embeddings
- faiss-cpu: For vector similarity search
- transformers: For Hugging Face models
- torch: For model computations
- langchain: For RAG pipeline (optional, can be removed if not used)
- python-dotenv: For environment variables (optional)
- pytest: For running tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

See LICENSE file.
