# 📄 PDF Parsing Support for Deep Researcher Agent

## 🎯 Overview

The Deep Researcher Agent now supports **PDF document parsing**! You can upload PDF files directly through the Streamlit interface or process them via the command line. The system uses multiple PDF parsing libraries to ensure maximum compatibility and reliability.

## 🚀 Features

### ✅ **Multi-Library Support**
- **PyMuPDF (fitz)**: Fast, reliable parsing with metadata extraction
- **pdfplumber**: Excellent for text extraction and table detection
- **PyPDF2**: Lightweight, basic PDF processing

### ✅ **Comprehensive PDF Processing**
- **Text Extraction**: Full document text with page-by-page breakdown
- **Metadata Extraction**: Title, author, creation date, page count, etc.
- **Table Detection**: Extract tables from PDF documents (pdfplumber)
- **Error Handling**: Graceful fallback between parsing libraries
- **Progress Tracking**: Real-time processing feedback

### ✅ **Integration**
- **Streamlit Interface**: Drag & drop PDF uploads
- **CLI Support**: Command-line PDF processing
- **Document Management**: Automatic indexing and vectorization
- **Research Queries**: PDF content searchable through natural language

## 📦 Installation

### Required Libraries

Install the PDF parsing libraries:

```bash
# Install all PDF libraries (recommended)
pip install PyPDF2 pdfplumber pymupdf

# Or install individually
pip install PyPDF2      # Basic PDF processing
pip install pdfplumber  # Advanced text and table extraction
pip install pymupdf     # Fast, comprehensive PDF parsing
```

### Verify Installation

```bash
python test_pdf_parsing.py
```

## 🌐 Streamlit Interface Usage

### 1. **Upload PDF Files**
1. Go to the **📁 Document Management** tab
2. Click **"Upload documents"**
3. Select PDF files (supports multiple files)
4. Click **"Process Uploaded Files"**

### 2. **View Processing Results**
- **Document Details**: Title, page count, parser used
- **Metadata**: Author, creation date, subject, etc.
- **Content Length**: Character count and processing stats
- **Error Handling**: Clear error messages for failed files

### 3. **Research PDF Content**
1. Go to **🔍 Research Query** tab
2. Ask questions about your PDF content
3. Get answers with source attribution
4. Export research reports including PDF sources

## 💻 Command Line Usage

### Index PDF Files

```bash
# Index a single PDF file
python cli.py index document.pdf --type file

# Index all PDFs in a directory
python cli.py index ./pdf_documents --type directory

# Index specific file formats
python cli.py index ./documents --formats pdf txt md
```

### Process PDFs Programmatically

```python
from utils.document_processor import DocumentProcessor

# Process a single PDF file
doc = DocumentProcessor.process_pdf_file("document.pdf")

# Process PDF data from bytes
with open("document.pdf", "rb") as f:
    pdf_data = f.read()
doc = DocumentProcessor.process_pdf_data(pdf_data, "document.pdf")
```

## 🔧 PDF Parser Configuration

### Parser Priority

The system tries parsers in this order:
1. **PyMuPDF** (fastest, most reliable)
2. **pdfplumber** (best for complex layouts)
3. **PyPDF2** (lightweight fallback)

### Custom Parser Usage

```python
from utils.pdf_parser import PDFParser

parser = PDFParser()
print(f"Available parsers: {parser.available_parsers}")

# Parse PDF with specific parser
result = parser.parse_pdf(pdf_data, "document.pdf")

# Extract tables (requires pdfplumber)
tables = parser.extract_tables(pdf_data)

# Get document info without full parsing
info = parser.get_document_info(pdf_data)
```

## 📊 Supported PDF Features

### ✅ **Text Extraction**
- Full document text
- Page-by-page breakdown
- Preserved formatting where possible
- Unicode support

### ✅ **Metadata Extraction**
- Document title and author
- Creation and modification dates
- Subject and keywords
- Creator and producer information
- Page count and file size

### ✅ **Table Detection** (pdfplumber)
- Automatic table identification
- Structured data extraction
- Page and table numbering
- Row and column counts

### ✅ **Error Handling**
- Graceful fallback between parsers
- Detailed error messages
- Partial content recovery
- Processing statistics

## 🎯 Use Cases

### 📚 **Academic Research**
- Process research papers and articles
- Extract citations and references
- Search across multiple documents
- Generate research summaries

### 📋 **Business Documents**
- Process reports and presentations
- Extract key information and data
- Search contract terms and conditions
- Analyze document collections

### 📖 **Knowledge Management**
- Build searchable document libraries
- Extract insights from PDF archives
- Create comprehensive research databases
- Generate automated summaries

## 🔍 Research Query Examples

Once PDFs are indexed, you can ask questions like:

```
"What are the main findings in the research papers?"
"Summarize the methodology used in the studies"
"What are the key recommendations from the reports?"
"Compare the results across different documents"
"Find information about specific topics or concepts"
```

## 📈 Performance Considerations

### **File Size Limits**
- **Small PDFs** (< 10MB): Fast processing
- **Medium PDFs** (10-50MB): Moderate processing time
- **Large PDFs** (> 50MB): May take longer, consider splitting

### **Processing Speed**
- **PyMuPDF**: Fastest, best for large files
- **pdfplumber**: Slower but better text quality
- **PyPDF2**: Fastest but basic functionality

### **Memory Usage**
- PDFs are processed in memory
- Large files may require more RAM
- Consider processing files individually for very large documents

## 🚨 Troubleshooting

### Common Issues

1. **"No PDF parsing libraries available"**
   ```bash
   pip install PyPDF2 pdfplumber pymupdf
   ```

2. **"Failed to parse PDF"**
   - Try a different PDF file
   - Check if the PDF is password-protected
   - Verify the file isn't corrupted

3. **"Empty content extracted"**
   - PDF might be image-based (scanned)
   - Try OCR preprocessing
   - Check if text is embedded as images

4. **"Memory error"**
   - Process smaller files
   - Increase system memory
   - Use PyMuPDF for better memory efficiency

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed parsing information
```

## 🎉 Success Indicators

You'll know PDF parsing is working when you see:
- ✅ PDF files upload successfully in Streamlit
- 📊 Document details show page count and metadata
- 🔍 Research queries return relevant PDF content
- 📄 Exported reports include PDF sources
- 📈 Analytics show PDF documents in knowledge base

## 🔗 Integration Examples

### Streamlit Integration
```python
# In your Streamlit app
uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'])
for file in uploaded_files:
    doc = DocumentProcessor.process_pdf_data(file.read(), file.name)
    # Process and index the document
```

### CLI Integration
```bash
# Batch process PDF directory
python cli.py index ./research_papers --formats pdf

# Process single PDF
python cli.py index paper.pdf --type file
```

### Python API Integration
```python
from researcher_agent import DeepResearcherAgent

agent = DeepResearcherAgent()
doc = DocumentProcessor.process_pdf_file("document.pdf")
agent.index_documents([doc], "pdf_knowledge_base")
result = agent.research_query("What is this document about?")
```

## 📚 Additional Resources

- **PyMuPDF Documentation**: https://pymupdf.readthedocs.io/
- **pdfplumber Documentation**: https://github.com/jsvine/pdfplumber
- **PyPDF2 Documentation**: https://pypdf2.readthedocs.io/
- **Streamlit File Upload**: https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader

---

**🎉 PDF parsing is now fully integrated into the Deep Researcher Agent!** Upload your PDFs and start researching with the power of local AI processing! 🚀
