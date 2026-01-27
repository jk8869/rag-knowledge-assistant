from pypdf import PdfReader

def extract_text_from_pdf(file_file):
    reader = PdfReader(file_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text