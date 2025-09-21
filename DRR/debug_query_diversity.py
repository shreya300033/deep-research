#!/usr/bin/env python3
"""
Debug script to test query diversity and search results
"""
import sys
import os
import numpy as np

def test_query_diversity():
    """Test if different queries produce different results"""
    print("🔍 Testing Query Diversity and Search Results")
    print("=" * 60)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        from embeddings.embedding_generator import EmbeddingGenerator
        
        # Initialize components
        print("🤖 Initializing components...")
        embedding_gen = EmbeddingGenerator()
        agent = DeepResearcherAgent(use_enhanced_reasoning=False)  # Use standard for debugging
        session_id = agent.start_research_session("diversity_test")
        print(f"✅ Session started: {session_id}")
        
        # Load sample documents
        print("📚 Loading sample documents...")
        sample_docs = DocumentProcessor.create_sample_documents()
        result = agent.index_documents(sample_docs, "diversity_test")
        print(f"✅ Indexed {result['documents_indexed']} documents")
        
        # Test different queries
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?", 
            "What are neural networks?",
            "Explain deep learning",
            "What is natural language processing?"
        ]
        
        print("\n🧪 Testing Query Embeddings...")
        embeddings = []
        for i, query in enumerate(test_queries):
            embedding = embedding_gen.generate_embedding(query)
            embeddings.append(embedding)
            print(f"Query {i+1}: {query}")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding sample: {embedding[:5]}")
            print()
        
        # Check if embeddings are different
        print("🔍 Checking Embedding Diversity...")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"Similarity between query {i+1} and {j+1}: {similarity:.4f}")
        
        print("\n🔍 Testing Search Results...")
        for i, query in enumerate(test_queries):
            print(f"\n--- Query {i+1}: {query} ---")
            
            # Get embedding and search
            query_embedding = embedding_gen.generate_embedding(query)
            search_results = agent.vector_store.search(query_embedding, k=5)
            
            print(f"Search results count: {len(search_results)}")
            
            if search_results:
                print("Top 3 results:")
                for j, result in enumerate(search_results[:3]):
                    print(f"  {j+1}. Source: {result.get('source', 'Unknown')}")
                    print(f"     Similarity: {result.get('similarity_score', 0):.4f}")
                    print(f"     Chunk preview: {result.get('chunk_text', '')[:100]}...")
                    print()
            else:
                print("  No results found!")
        
        print("\n🔍 Testing Full Query Processing...")
        for i, query in enumerate(test_queries):
            print(f"\n--- Full Query {i+1}: {query} ---")
            
            result = agent.research_query(query, max_reasoning_steps=2)
            
            print(f"Query type: {result.get('query_type', 'unknown')}")
            print(f"Reasoning steps: {len(result.get('reasoning_steps', []))}")
            
            # Show first step answer
            steps = result.get('reasoning_steps', [])
            if steps:
                first_step = steps[0]
                print(f"First step answer: {first_step.get('answer', 'No answer')[:200]}...")
                print(f"Sources: {first_step.get('sources', [])}")
            else:
                print("No reasoning steps found!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_directly():
    """Test vector store directly"""
    print("\n🔍 Testing Vector Store Directly")
    print("=" * 40)
    
    try:
        from researcher_agent import DeepResearcherAgent
        from utils.document_processor import DocumentProcessor
        from embeddings.embedding_generator import EmbeddingGenerator
        
        # Initialize
        embedding_gen = EmbeddingGenerator()
        agent = DeepResearcherAgent()
        agent.start_research_session("direct_test")
        
        # Load documents
        sample_docs = DocumentProcessor.create_sample_documents()
        agent.index_documents(sample_docs, "direct_test")
        
        # Check vector store stats
        stats = agent.get_vector_store_stats()
        print(f"Vector store stats: {stats}")
        
        # Test different queries
        queries = ["AI", "machine learning", "neural networks", "deep learning"]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            embedding = embedding_gen.generate_embedding(query)
            results = agent.vector_store.search(embedding, k=3)
            
            print(f"Results: {len(results)}")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.get('source', 'Unknown')} - {result.get('similarity_score', 0):.3f}")
                print(f"     {result.get('chunk_text', '')[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run diagnostic tests"""
    print("🔧 Query Diversity Diagnostic Tool")
    print("=" * 50)
    
    success1 = test_query_diversity()
    success2 = test_vector_store_directly()
    
    if success1 and success2:
        print("\n✅ Diagnostic tests completed!")
        print("Check the output above to identify why queries return the same results.")
    else:
        print("\n❌ Some diagnostic tests failed.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
