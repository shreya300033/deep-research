"""
Simple test script to verify the Deep Researcher Agent system
"""
import sys
import traceback

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from researcher_agent import DeepResearcherAgent
        print("✓ DeepResearcherAgent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DeepResearcherAgent: {e}")
        return False
    
    try:
        from embeddings.embedding_generator import EmbeddingGenerator
        print("✓ EmbeddingGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import EmbeddingGenerator: {e}")
        return False
    
    try:
        from retrieval.vector_store import VectorStore
        print("✓ VectorStore imported successfully")
    except Exception as e:
        print(f"✗ Failed to import VectorStore: {e}")
        return False
    
    try:
        from reasoning.reasoning_engine import ReasoningEngine
        print("✓ ReasoningEngine imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ReasoningEngine: {e}")
        return False
    
    try:
        from utils.document_processor import DocumentProcessor
        print("✓ DocumentProcessor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DocumentProcessor: {e}")
        return False
    
    try:
        from export.pdf_exporter import PDFExporter
        print("✓ PDFExporter imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PDFExporter: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Initialize agent
        agent = DeepResearcherAgent()
        print("✓ Agent initialized successfully")
        
        # Start session
        session_id = agent.start_research_session("test_session")
        print(f"✓ Research session started: {session_id}")
        
        # Create sample documents
        sample_docs = DocumentProcessor.create_sample_documents()
        print(f"✓ Created {len(sample_docs)} sample documents")
        
        # Index documents
        result = agent.index_documents(sample_docs, "test_index")
        print(f"✓ Documents indexed: {result}")
        
        # Test simple query
        query_result = agent.research_query("What is artificial intelligence?")
        print(f"✓ Query processed successfully")
        print(f"  - Query type: {query_result.get('query_type', 'unknown')}")
        print(f"  - Reasoning steps: {len(query_result.get('reasoning_steps', []))}")
        print(f"  - Total sources: {query_result.get('total_sources', 0)}")
        
        # Test export
        report_content = agent.export_research_report(session_id, "markdown")
        print(f"✓ Report exported successfully (length: {len(report_content)} chars)")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that required dependencies are available"""
    print("\nTesting dependencies...")
    
    dependencies = [
        'sentence_transformers',
        'numpy',
        'pandas',
        'sklearn',
        'faiss',
        'transformers',
        'torch',
        'tiktoken',
        'reportlab'
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} missing")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=== Deep Researcher Agent System Test ===\n")
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please install missing dependencies.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check for syntax errors in the code.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed.")
        return False
    
    print("\n✅ All tests passed! The Deep Researcher Agent system is working correctly.")
    print("\nYou can now run 'python example_usage.py' to see the full demo.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
