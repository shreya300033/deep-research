"""
PDF parsing utilities for the Deep Researcher Agent
"""
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParser:
    """Handles PDF document parsing with multiple backends"""
    
    def __init__(self):
        self.available_parsers = self._check_available_parsers()
        logger.info(f"Available PDF parsers: {self.available_parsers}")
    
    def _check_available_parsers(self) -> List[str]:
        """Check which PDF parsing libraries are available"""
        parsers = []
        
        try:
            import PyPDF2
            parsers.append("PyPDF2")
        except ImportError:
            pass
        
        try:
            import pdfplumber
            parsers.append("pdfplumber")
        except ImportError:
            pass
        
        try:
            import fitz  # PyMuPDF
            parsers.append("pymupdf")
        except ImportError:
            pass
        
        return parsers
    
    def parse_pdf(self, pdf_data: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """Parse PDF data and extract text content"""
        if not self.available_parsers:
            raise ImportError("No PDF parsing libraries available. Please install PyPDF2, pdfplumber, or pymupdf.")
        
        # Try parsers in order of preference
        for parser in ["pymupdf", "pdfplumber", "PyPDF2"]:
            if parser in self.available_parsers:
                try:
                    if parser == "pymupdf":
                        return self._parse_with_pymupdf(pdf_data, filename)
                    elif parser == "pdfplumber":
                        return self._parse_with_pdfplumber(pdf_data, filename)
                    elif parser == "PyPDF2":
                        return self._parse_with_pypdf2(pdf_data, filename)
                except Exception as e:
                    logger.warning(f"Failed to parse with {parser}: {e}")
                    continue
        
        raise Exception("All PDF parsing methods failed")
    
    def _parse_with_pymupdf(self, pdf_data: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF (fitz)"""
        import fitz
        
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text_content = []
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'page_count': doc.page_count
        }
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text_content.append({
                    'page_number': page_num + 1,
                    'text': page_text.strip()
                })
        
        doc.close()
        
        full_text = '\n\n'.join([page['text'] for page in text_content])
        
        return {
            'filename': filename,
            'full_text': full_text,
            'pages': text_content,
            'metadata': metadata,
            'parser_used': 'pymupdf',
            'total_pages': len(text_content),
            'total_characters': len(full_text)
        }
    
    def _parse_with_pdfplumber(self, pdf_data: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using pdfplumber"""
        import pdfplumber
        
        text_content = []
        metadata = {}
        
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            # Extract metadata
            if pdf.metadata:
                metadata = {
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'subject': pdf.metadata.get('Subject', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                    'producer': pdf.metadata.get('Producer', ''),
                    'creation_date': pdf.metadata.get('CreationDate', ''),
                    'modification_date': pdf.metadata.get('ModDate', ''),
                    'page_count': len(pdf.pages)
                }
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append({
                        'page_number': page_num + 1,
                        'text': page_text.strip()
                    })
        
        full_text = '\n\n'.join([page['text'] for page in text_content])
        
        return {
            'filename': filename,
            'full_text': full_text,
            'pages': text_content,
            'metadata': metadata,
            'parser_used': 'pdfplumber',
            'total_pages': len(text_content),
            'total_characters': len(full_text)
        }
    
    def _parse_with_pypdf2(self, pdf_data: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using PyPDF2"""
        import PyPDF2
        
        text_content = []
        metadata = {}
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        
        # Extract metadata
        if pdf_reader.metadata:
            metadata = {
                'title': pdf_reader.metadata.get('/Title', ''),
                'author': pdf_reader.metadata.get('/Author', ''),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creator': pdf_reader.metadata.get('/Creator', ''),
                'producer': pdf_reader.metadata.get('/Producer', ''),
                'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                'modification_date': str(pdf_reader.metadata.get('/ModDate', '')),
                'page_count': len(pdf_reader.pages)
            }
        
        # Extract text from each page
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append({
                        'page_number': page_num + 1,
                        'text': page_text.strip()
                    })
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        full_text = '\n\n'.join([page['text'] for page in text_content])
        
        return {
            'filename': filename,
            'full_text': full_text,
            'pages': text_content,
            'metadata': metadata,
            'parser_used': 'PyPDF2',
            'total_pages': len(text_content),
            'total_characters': len(full_text)
        }
    
    def extract_tables(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract tables from PDF (requires pdfplumber)"""
        if "pdfplumber" not in self.available_parsers:
            logger.warning("Table extraction requires pdfplumber")
            return []
        
        import pdfplumber
        
        tables = []
        
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table_num, table in enumerate(page_tables):
                    if table:
                        tables.append({
                            'page_number': page_num + 1,
                            'table_number': table_num + 1,
                            'data': table,
                            'rows': len(table),
                            'columns': len(table[0]) if table else 0
                        })
        
        return tables
    
    def get_document_info(self, pdf_data: bytes) -> Dict[str, Any]:
        """Get basic document information without full parsing"""
        if not self.available_parsers:
            return {"error": "No PDF parsing libraries available"}
        
        # Use the first available parser for basic info
        parser = self.available_parsers[0]
        
        try:
            if parser == "pymupdf":
                import fitz
                doc = fitz.open(stream=pdf_data, filetype="pdf")
                info = {
                    'page_count': doc.page_count,
                    'metadata': doc.metadata,
                    'file_size': len(pdf_data)
                }
                doc.close()
                return info
            elif parser == "pdfplumber":
                import pdfplumber
                with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                    return {
                        'page_count': len(pdf.pages),
                        'metadata': pdf.metadata or {},
                        'file_size': len(pdf_data)
                    }
            elif parser == "PyPDF2":
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                return {
                    'page_count': len(pdf_reader.pages),
                    'metadata': pdf_reader.metadata or {},
                    'file_size': len(pdf_data)
                }
        except Exception as e:
            return {"error": f"Failed to get document info: {e}"}
        
        return {"error": "Unknown parser"}

def process_pdf_file(file_path: str) -> Dict[str, Any]:
    """Process a PDF file and return document data"""
    parser = PDFParser()
    
    try:
        with open(file_path, 'rb') as f:
            pdf_data = f.read()
        
        result = parser.parse_pdf(pdf_data, Path(file_path).name)
        
        # Convert to document format expected by the system
        document = {
            'id': f"pdf_{Path(file_path).stem}",
            'title': result['metadata'].get('title') or Path(file_path).stem,
            'content': result['full_text'],
            'source': file_path,
            'type': 'pdf',
            'metadata': result['metadata'],
            'page_count': result['total_pages'],
            'parser_used': result['parser_used']
        }
        
        return document
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        return {
            'id': f"pdf_{Path(file_path).stem}",
            'title': Path(file_path).stem,
            'content': f"Error processing PDF: {e}",
            'source': file_path,
            'type': 'pdf',
            'error': str(e)
        }

def process_pdf_data(pdf_data: bytes, filename: str) -> Dict[str, Any]:
    """Process PDF data and return document data"""
    parser = PDFParser()
    
    try:
        result = parser.parse_pdf(pdf_data, filename)
        
        # Convert to document format expected by the system
        document = {
            'id': f"pdf_{Path(filename).stem}",
            'title': result['metadata'].get('title') or Path(filename).stem,
            'content': result['full_text'],
            'source': filename,
            'type': 'pdf',
            'metadata': result['metadata'],
            'page_count': result['total_pages'],
            'parser_used': result['parser_used']
        }
        
        return document
        
    except Exception as e:
        logger.error(f"Error processing PDF data {filename}: {e}")
        return {
            'id': f"pdf_{Path(filename).stem}",
            'title': Path(filename).stem,
            'content': f"Error processing PDF: {e}",
            'source': filename,
            'type': 'pdf',
            'error': str(e)
        }
