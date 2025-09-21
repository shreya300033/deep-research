# Deep Researcher Agent

A comprehensive Python-based system for deep research that can search, analyze, and synthesize information from large-scale data sources without relying on external web search APIs. The system handles local embedding generation and reasoning to provide intelligent research capabilities.

## Features

### Core Requirements ✅
- **Python-based system** for query handling and response generation
- **Local embedding generation** for document indexing and retrieval using sentence-transformers
- **Multi-step reasoning** to break down complex queries into smaller, manageable tasks
- **Efficient storage and retrieval pipeline** using FAISS for vector similarity search

### Enhanced Features ✅
- **Summarization** of multiple sources into coherent research reports
- **Interactive query refinement** with follow-up question suggestions
- **AI-powered assistant** that explains reasoning steps and provides confidence scores
- **Export functionality** for research results in PDF and Markdown formats

## System Architecture

```
Deep Researcher Agent
├── Embedding Generation (sentence-transformers)
├── Vector Storage (FAISS)
├── Multi-step Reasoning Engine
├── Query Analysis & Decomposition
├── Document Processing
└── Export System (PDF/Markdown)
```

## Installation

1. **Clone or download the project files**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python example_usage.py
   ```

## Quick Start

### Basic Usage

```python
from researcher_agent import DeepResearcherAgent
from utils.document_processor import DocumentProcessor

# Initialize the agent
agent = DeepResearcherAgent()

# Start a research session
session_id = agent.start_research_session("My Research")

# Index documents (using sample data)
sample_docs = DocumentProcessor.create_sample_documents()
agent.index_documents(sample_docs, "knowledge_base")

# Process a research query
result = agent.research_query("What is artificial intelligence?")

# Export results
agent.save_pdf_report(session_id, "my_research_report")
```

### Advanced Usage

```python
# Complex analytical query with multi-step reasoning
complex_query = "Compare machine learning approaches and their applications in computer vision"
result = agent.research_query(complex_query, max_reasoning_steps=6)

# Query refinement
refined_result = agent.refine_query(
    "What is machine learning?", 
    "Focus specifically on deep learning and neural networks"
)

# Get research summary
summary = agent.get_research_summary(session_id)
```

## Components

### 1. Embedding Generator (`embeddings/embedding_generator.py`)
- Uses sentence-transformers for local embedding generation
- Supports text chunking with configurable overlap
- Handles embedding storage and retrieval

### 2. Vector Store (`retrieval/vector_store.py`)
- FAISS-based vector storage for efficient similarity search
- Supports metadata storage and retrieval
- Configurable similarity thresholds

### 3. Reasoning Engine (`reasoning/reasoning_engine.py`)
- Multi-step reasoning with query decomposition
- Confidence scoring for reasoning steps
- Context-aware synthesis

### 4. Query Analyzer (`reasoning/query_analyzer.py`)
- Classifies query types (factual, analytical, comparative, etc.)
- Decomposes complex queries into reasoning steps
- Generates follow-up questions

### 5. Document Processor (`utils/document_processor.py`)
- Handles various document formats (text, JSON, markdown)
- Batch processing capabilities
- Sample data generation

### 6. PDF Exporter (`export/pdf_exporter.py`)
- Professional PDF report generation
- Structured formatting with tables and styles
- Source attribution and metadata

## Configuration

Edit `config.py` to customize:

```python
# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.7

# Reasoning settings
MAX_REASONING_STEPS = 5
```

## File Structure

```
DRR/
├── researcher_agent.py          # Main agent class
├── config.py                   # Configuration settings
├── requirements.txt            # Dependencies
├── example_usage.py           # Usage examples
├── README.md                  # This file
├── embeddings/
│   └── embedding_generator.py # Local embedding generation
├── retrieval/
│   └── vector_store.py        # FAISS vector storage
├── reasoning/
│   ├── query_analyzer.py      # Query analysis & decomposition
│   └── reasoning_engine.py    # Multi-step reasoning
├── utils/
│   └── document_processor.py  # Document processing utilities
├── export/
│   └── pdf_exporter.py        # PDF export functionality
├── data/                      # Input documents (created automatically)
├── embeddings/                # Stored embeddings (created automatically)
├── index/                     # FAISS indices (created automatically)
└── reports/                   # Generated reports (created automatically)
```

## API Reference

### DeepResearcherAgent

#### Core Methods
- `start_research_session(session_name)` - Start a new research session
- `index_documents(documents, index_name)` - Index documents for research
- `research_query(query, max_reasoning_steps)` - Process a research query
- `refine_query(original_query, refinement)` - Refine a previous query

#### Export Methods
- `export_research_report(session_id, format)` - Export in markdown/json/pdf
- `save_pdf_report(session_id, filename)` - Save PDF report directly
- `save_report(content, filename, format)` - Save report to file

#### Utility Methods
- `get_research_summary(session_id)` - Get session summary
- `get_vector_store_stats()` - Get indexing statistics
- `load_index(index_name)` - Load existing index
- `clear_research_history()` - Clear research history

## Query Types Supported

1. **Factual Queries** - "What is X?", "Who is Y?"
2. **Analytical Queries** - "Analyze X", "Evaluate Y"
3. **Comparative Queries** - "Compare X and Y", "Difference between A and B"
4. **Causal Queries** - "Why does X happen?", "What causes Y?"
5. **Procedural Queries** - "How to do X?", "Steps for Y"
6. **Conceptual Queries** - "Explain the concept of X", "Theory of Y"

## Performance Considerations

- **Embedding Model**: Uses `all-MiniLM-L6-v2` (384 dimensions) for fast processing
- **Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Indexing**: FAISS provides fast similarity search even with large document collections
- **Memory**: Efficient storage of embeddings and metadata

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Memory Issues**: Reduce chunk size or use smaller embedding models
3. **Slow Performance**: Consider using GPU-accelerated sentence-transformers

### Dependencies

- Python 3.8+
- sentence-transformers
- faiss-cpu
- numpy, pandas, scikit-learn
- reportlab (for PDF export)
- transformers, torch

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic research queries
- Complex analytical questions
- Interactive query refinement
- Report generation and export
- Advanced multi-step reasoning

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to extend the system with additional features:
- Support for more document formats
- Additional embedding models
- Enhanced reasoning strategies
- Web interface
- Database integration
