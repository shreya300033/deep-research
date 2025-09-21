"""
Configuration settings for the Deep Researcher Agent
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
REPORTS_DIR = BASE_DIR / "reports"
INDEX_DIR = BASE_DIR / "index"

# Create directories if they don't exist
for directory in [DATA_DIR, EMBEDDINGS_DIR, REPORTS_DIR, INDEX_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model
CHUNK_SIZE = 1000  # Text chunk size for processing
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_TOKENS = 4000  # Maximum tokens for processing

# Retrieval settings
TOP_K_RESULTS = 15  # Number of top results to retrieve
SIMILARITY_THRESHOLD = 0.1  # Minimum similarity score (very low for better recall)

# Reasoning settings
MAX_REASONING_STEPS = 5  # Maximum steps for multi-step reasoning
TEMPERATURE = 0.7  # Temperature for text generation

# Export settings
PDF_TEMPLATE = "research_report_template.html"
MARKDOWN_TEMPLATE = "research_report_template.md"