from typing import Union, List, Literal
import glob
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Set up logging to capture errors, warnings, and info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_non_utf8_characters(text):
    return ''.join(char for char in text if ord(char) < 128)

def load_pdf(pdf_file):
    try:
        docs = PyPDFLoader(pdf_file, extract_images=True).load()
        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return docs
    except Exception as e:
        logger.error(f"Failed to load PDF {pdf_file}: {str(e)}")
        return []  # Return empty list for failed PDFs to continue processing

def is_likely_pdf(file_path):
    """Check if a file is likely a PDF by reading its header."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            return header.startswith(b'%PDF-')
    except Exception:
        return False

class BaseLoader:
    def __init__(self) -> None:
        pass

    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    """Loads and cleans PDF documents sequentially."""
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        doc_loaded = []
        total_files = len(pdf_files)
        with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
            for pdf_file in pdf_files:
                if not is_likely_pdf(pdf_file):
                    logger.warning(f"Skipping {pdf_file}: Not a valid PDF file")
                    pbar.update(1)
                    continue
                result = load_pdf(pdf_file)  # Load each PDF sequentially
                if result:  # Check if documents were successfully loaded
                    logger.info(f"Successfully loaded PDF {pdf_file}")
                    doc_loaded.extend(result)  # Merge results into one list
                pbar.update(1)  # Update progress bar
        return doc_loaded

class TextSplitter:
    def __init__(self,
                 separators: List[str] = ['\n\n', '\n', '', ''],
                 chunk_size: int = 300,
                 chunk_overlap: int = 50
                 ) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    def __call__(self, documents):
        return self.splitter.split_documents(documents)

class Loader:
    def __init__(self,
                 file_type: str = Literal["pdf"],
                 split_kwargs: dict = {"chunk_size": 300, "chunk_overlap": 50}
                 ) -> None:
        
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.doc_spltter = TextSplitter(**split_kwargs)
    
    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        # Ignore workers since we're not using multiprocessing
        doc_loaded = self.doc_loader(pdf_files)
        doc_split = self.doc_spltter(doc_loaded)
        return doc_split
    
    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=workers)