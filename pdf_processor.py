import pdfplumber
from config import CHUNK_SIZE
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF file, returning a list of dictionaries with page number and text.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list[dict]: List of {'page': int, 'text': str} for each page.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append({'page': page_num, 'text': text.strip()})
    return pages_text

def split_text_into_chunks(text_list: list[dict], chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Splits the extracted text into smaller chunks for indexing.

    Args:
        text_list (list[dict]): List from extract_text_from_pdf.
        chunk_size (int): Approximate number of characters per chunk.

    Returns:
        list[dict]: List of {'page': int, 'text': str, 'chunk_id': int}
    """
    chunks = []
    chunk_id = 0
    for page_data in text_list:
        text = page_data['text']
        page = page_data['page']
        words = text.split()
        current_chunk = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({'page': page, 'text': chunk_text, 'chunk_id': chunk_id})
                chunk_id += 1
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({'page': page, 'text': chunk_text, 'chunk_id': chunk_id})
            chunk_id += 1
    return chunks
