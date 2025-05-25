import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def is_valid_pdf(file_path: str) -> bool:
    """
    Check if a file is a valid PDF by verifying the '%PDF-' header.

    :param file_path: Path to the file.
    :return: True if the file is a valid PDF, False otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8).decode('utf-8', errors='ignore')
            return header.startswith('%PDF-')
    except Exception as e:
        logger.error(f"Error checking {file_path}: {str(e)}")
        return False

def remove_invalid_pdfs(directory: str) -> None:
    """
    Move invalid PDF files from the specified directory to an 'invalid' subdirectory.

    :param directory: Path to the directory containing PDF files.
    """
    # Convert to Path object for easier handling
    dir_path = Path(directory)
    
    # Create 'invalid' subdirectory if it doesn't exist
    invalid_dir = dir_path / 'invalid'
    invalid_dir.mkdir(exist_ok=True)
    
    # Get all .pdf files in the directory
    pdf_files = [f for f in dir_path.glob('*.pdf') if f.is_file()]
    
    if not pdf_files:
        logger.info(f"No PDF files found in {directory}")
        return
    
    # Process each PDF file
    for pdf_file in pdf_files:
        if is_valid_pdf(pdf_file):
            logger.info(f"Valid PDF: {pdf_file}")
        else:
            logger.warning(f"Invalid PDF: {pdf_file}. Moving to {invalid_dir}")
            # Move the invalid PDF to the 'invalid' subdirectory
            try:
                pdf_file.rename(invalid_dir / pdf_file.name)
            except Exception as e:
                logger.error(f"Failed to move {pdf_file}: {str(e)}")

if __name__ == '__main__':
    # Specify the directory containing the PDFs
    directory = './data_source/generative_ai/data'
    
    # Verify the directory exists
    if not Path(directory).is_dir():
        logger.error(f"Directory {directory} does not exist")
    else:
        logger.info(f"Processing PDFs in {directory}")
        remove_invalid_pdfs(directory)