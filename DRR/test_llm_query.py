#!/usr/bin/env python3
"""
Test script for LLM query to debug the issue
"""
import sys
import os

def test_llm_query():
    """Test the LLM query specifically"""
    print("🔍 Testing LLM Query - Debug Mode")
    print("=" * 50)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        
        # Initialize agent
        print("🤖 Initializing agent...")
        agent = DeepResearcherAgent(use_enhanced_reasoning=True)
        session_id = agent.start_research_session("llm_test")
        print(f"✅ Session started: {session_id}")
        
        # Check if documents are already indexed
        stats = agent.get_vector_store_stats()
        print(f"📊 Current vector store stats: {stats}")
        
        if stats.get('total_vectors', 0) == 0:
            print("📚 No documents indexed. Loading sample documents...")
            sample_docs = DocumentProcessor.create_sample_documents()
            result = agent.index_documents(sample_docs, "llm_test")
            print(f"✅ Indexed {result['documents_indexed']} documents")
            
            # Check stats again
            stats = agent.get_vector_store_stats()
            print(f"📊 Updated vector store stats: {stats}")
        else:
            print("📚 Documents already indexed")
        
        # Test the specific LLM query
        query = "what is llm"
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test with standard reasoning first
        print("\n--- Standard Reasoning ---")
        agent_standard = DeepResearcherAgent(use_enhanced_reasoning=False)
        agent_standard.start_research_session("standard_test")
        agent_standard.index_documents(sample_docs, "standard_test")
        
        standard_result = agent_standard.research_query(query, max_reasoning_steps=3)
        print(f"Standard reasoning result:")
        print(f"  Query type: {standard_result.get('query_type', 'unknown')}")
        print(f"  Reasoning steps: {len(standard_result.get('reasoning_steps', []))}")
        
        steps = standard_result.get('reasoning_steps', [])
        for i, step in enumerate(steps, 1):
            print(f"  Step {i}: {step.get('description', 'No description')}")
            print(f"    Answer: {step.get('answer', 'No answer')[:100]}...")
            print(f"    Sources: {step.get('sources', [])}")
        
        # Test with enhanced reasoning
        print("\n--- Enhanced Reasoning ---")
        enhanced_result = agent.research_query(query, max_reasoning_steps=3)
        print(f"Enhanced reasoning result:")
        print(f"  Query type: {enhanced_result.get('query_type', 'unknown')}")
        print(f"  Reasoning steps: {len(enhanced_result.get('reasoning_steps', []))}")
        
        steps = enhanced_result.get('reasoning_steps', [])
        for i, step in enumerate(steps, 1):
            print(f"  Step {i}: {step.get('description', 'No description')}")
            print(f"    Answer: {step.get('answer', 'No answer')[:100]}...")
            print(f"    Sources: {step.get('sources', [])}")
            print(f"    Reasoning pattern: {step.get('reasoning_pattern', 'unknown')}")
        
        # Test direct vector search
        print("\n--- Direct Vector Search ---")
        from embeddings.embedding_generator import EmbeddingGenerator
        embedding_gen = EmbeddingGenerator()
        query_embedding = embedding_gen.generate_embedding(query)
        search_results = agent.vector_store.search(query_embedding, k=5)
        
        print(f"Direct search results: {len(search_results)}")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. Source: {result.get('source', 'Unknown')}")
            print(f"     Similarity: {result.get('similarity_score', 0):.4f}")
            print(f"     Chunk: {result.get('chunk_text', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run LLM query test"""
    print("🔍 LLM Query Debug Test")
    print("=" * 30)
    
    success = test_llm_query()
    
    if success:
        print("\n✅ LLM query test completed!")
    else:
        print("\n❌ LLM query test failed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
