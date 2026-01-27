from pypdf import PdfReader

def extract_text_from_pdf(file_file):
    reader = PdfReader(file_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Splits text into chunks of `chunk_size` characters, 
    with `overlap` characters shared between chunks to preserve context.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # Slice the text
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the start forward, subtracting overlap
        # (If we are at the end, just break)
        if end >= text_length:
            break

        start += (chunk_size - overlap)

    return chunks