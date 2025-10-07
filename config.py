import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

# Configuration settings
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
SUMMARIZATION_MODEL = os.getenv('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
DEFAULT_INDEX_PATH = os.getenv('DEFAULT_INDEX_PATH', 'index.pkl')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validate configurations
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be a positive integer")

if LOG_LEVEL not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
    raise ValueError("LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
