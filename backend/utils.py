from langchain.document_loaders import PyPDFLoader
from pathlib import Path

def load_pdf(file_path: Path):
    """Load PDF document"""
    try:
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")

def format_response(response: str) -> str:
    """Format the response"""
    return response.strip()