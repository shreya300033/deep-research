#!/usr/bin/env python3
"""
Simple test to verify the Deep Researcher Agent is working
"""
import sys
import os

print("=== Deep Researcher Agent Simple Test ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("\n1. Testing imports...")
    from researcher_agent import DeepResearcherAgent
    print("✓ DeepResearcherAgent imported successfully")
    
    from utils.document_processor import DocumentProcessor
    print("✓ DocumentProcessor imported successfully")
    
    print("\n2. Initializing agent...")
    agent = DeepResearcherAgent()
    print("✓ Agent initialized successfully")
    
    print("\n3. Starting research session...")
    session_id = agent.start_research_session("test_session")
    print(f"✓ Session started: {session_id}")
    
    print("\n4. Creating sample documents...")
    sample_docs = DocumentProcessor.create_sample_documents()
    print(f"✓ Created {len(sample_docs)} sample documents")
    
    print("\n5. Indexing documents...")
    result = agent.index_documents(sample_docs, "test_index")
    print(f"✓ Indexing result: {result}")
    
    print("\n6. Processing a simple query...")
    query_result = agent.research_query("What is artificial intelligence?")
    print(f"✓ Query processed successfully")
    print(f"  - Query type: {query_result.get('query_type', 'unknown')}")
    print(f"  - Reasoning steps: {len(query_result.get('reasoning_steps', []))}")
    print(f"  - Total sources: {query_result.get('total_sources', 0)}")
    
    print("\n7. Testing export...")
    report_content = agent.export_research_report(session_id, "markdown")
    print(f"✓ Report exported successfully (length: {len(report_content)} chars)")
    
    print("\n🎉 All tests passed! The Deep Researcher Agent is working correctly.")
    print("\nYou can now:")
    print("- Run 'python example_usage.py' for the full demo")
    print("- Use 'python cli.py demo --interactive' for interactive mode")
    print("- Check the README.md for detailed usage instructions")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
