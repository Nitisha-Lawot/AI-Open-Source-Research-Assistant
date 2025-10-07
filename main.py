import argparse
import sys
import os
import logging
import time
from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from indexer import SemanticIndexer
from query import ResearchAssistant
from config import LOG_LEVEL, DEFAULT_INDEX_PATH

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_pdf_paths(pdf_paths):
    """Validate that all provided paths are valid PDF files."""
    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")
        if not path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {path}")

def main():
    parser = argparse.ArgumentParser(description="AI Open Source Research Assistant")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest PDF files and build index')
    ingest_parser.add_argument('pdf_paths', nargs='+', help='Paths to PDF files')
    ingest_parser.add_argument('--index-path', default=DEFAULT_INDEX_PATH, help='Path to save the index file')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the assistant')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--index-path', default=DEFAULT_INDEX_PATH, help='Path to the index file')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of top results to retrieve')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--index-path', default=DEFAULT_INDEX_PATH, help='Path to the index file')
    benchmark_parser.add_argument('--num-queries', type=int, default=10, help='Number of queries to run for benchmarking')

    args = parser.parse_args()

    if args.command == 'ingest':
        logger.info(f"Starting ingestion of {len(args.pdf_paths)} PDF files")
        try:
            validate_pdf_paths(args.pdf_paths)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"PDF validation failed: {e}")
            print(f"Error: {e}")
            sys.exit(1)

        # Collect all texts and metadatas
        all_texts = []
        all_metadatas = []
        for pdf_path in args.pdf_paths:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                pages_text = extract_text_from_pdf(pdf_path)
                chunks = split_text_into_chunks(pages_text)
                for chunk in chunks:
                    all_texts.append(chunk['text'])
                    all_metadatas.append({'page': chunk['page'], 'chunk_id': chunk['chunk_id']})
                logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                print(f"Error processing {pdf_path}: {e}")
                sys.exit(1)

        if not all_texts:
            logger.error("No text extracted from PDFs")
            print("No text extracted from PDFs. Please check the files.")
            sys.exit(1)

        # Build or update index
        try:
            try:
                indexer = SemanticIndexer.load(args.index_path)
                logger.info(f"Loaded existing index from {args.index_path}")
                indexer.add_texts(all_texts, all_metadatas)
                logger.info(f"Added {len(all_texts)} new text chunks to index")
            except FileNotFoundError:
                indexer = SemanticIndexer()
                indexer.build_index(all_texts, all_metadatas)
                logger.info(f"Built new index with {len(all_texts)} text chunks")
            indexer.save(args.index_path)
            logger.info(f"Index saved to {args.index_path}")
            print(f"Index saved to {args.index_path}")
        except Exception as e:
            logger.error(f"Error processing index: {e}")
            print(f"Error processing index: {e}")
            sys.exit(1)

    elif args.command == 'query':
        logger.info(f"Processing query: '{args.question}' with top_k={args.top_k}")
        if not args.question.strip():
            logger.error("Question is empty")
            print("Error: Question cannot be empty.")
            sys.exit(1)

        if args.top_k <= 0:
            logger.error(f"Invalid top_k value: {args.top_k}")
            print("Error: top-k must be a positive integer.")
            sys.exit(1)

        # Load index
        try:
            logger.info(f"Loading index from {args.index_path}")
            indexer = SemanticIndexer.load(args.index_path)
        except FileNotFoundError:
            logger.error(f"Index file not found: {args.index_path}")
            print(f"Index file {args.index_path} not found. Please run 'ingest' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            print(f"Error loading index: {e}")
            sys.exit(1)

        # Query
        try:
            logger.info("Performing semantic search and generating answer")
            assistant = ResearchAssistant(indexer)
            result = assistant.answer_question(args.question, top_k=args.top_k)
            logger.info("Query completed successfully")

            print("Answer:", result['answer'])
            print("\nEvidence:")
            for i, ev in enumerate(result['evidence'], 1):
                print(f"{i}. Page {ev['page']}: {ev['text'][:100]}...")
        except Exception as e:
            logger.error(f"Error during query: {e}")
            print(f"Error during query: {e}")
            sys.exit(1)

    elif args.command == 'benchmark':
        logger.info(f"Starting benchmark with {args.num_queries} queries on index {args.index_path}")

        # Load index and measure time
        start_time = time.time()
        try:
            indexer = SemanticIndexer.load(args.index_path)
        except FileNotFoundError:
            logger.error(f"Index file not found: {args.index_path}")
            print(f"Index file {args.index_path} not found. Please run 'ingest' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            print(f"Error loading index: {e}")
            sys.exit(1)
        load_time = time.time() - start_time
        logger.info(f"Index loaded in {load_time:.2f} seconds")

        # Sample queries
        sample_questions = [
            "What is artificial intelligence?",
            "Explain machine learning.",
            "What are the benefits of AI?",
            "How does deep learning work?",
            "What is natural language processing?",
            "Describe computer vision.",
            "What are neural networks?",
            "Explain supervised learning.",
            "What is unsupervised learning?",
            "How to train a model?"
        ]

        # Run queries
        query_times = []
        assistant = ResearchAssistant(indexer)
        for i in range(min(args.num_queries, len(sample_questions))):
            question = sample_questions[i]
            start_time = time.time()
            try:
                result = assistant.answer_question(question, top_k=5)
                query_time = time.time() - start_time
                query_times.append(query_time)
                logger.info(f"Query {i+1}: {query_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error during benchmark query {i+1}: {e}")
                query_times.append(float('inf'))

        # Calculate stats
        valid_times = [t for t in query_times if t != float('inf')]
        if valid_times:
            avg_query_time = sum(valid_times) / len(valid_times)
            min_query_time = min(valid_times)
            max_query_time = max(valid_times)
            print(f"Benchmark Results:")
            print(f"Index load time: {load_time:.2f} seconds")
            print(f"Average query time: {avg_query_time:.2f} seconds")
            print(f"Min query time: {min_query_time:.2f} seconds")
            print(f"Max query time: {max_query_time:.2f} seconds")
            print(f"Successful queries: {len(valid_times)}/{args.num_queries}")
        else:
            print("No successful queries during benchmark.")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
