"""
Demo script for the Universal RAG Pipeline.

This script demonstrates how to use the RAG pipeline for:
1. Ingesting documents
2. Indexing them into a vector store
3. Querying the indexed documents
"""

import os
import sys
from pathlib import Path

from pipeline import RAGPipeline, create_pipeline
from utils import RAGConfig, EmbeddingConfig, LLMConfig, ChunkingConfig, VectorStoreConfig, RetrievalConfig


def example_1_quick_start():
    """
    Example 1: Quick start with default configuration.

    Uses OpenAI for embeddings and LLM with Chroma vector store.
    """
    print("=" * 80)
    print("EXAMPLE 1: Quick Start with Defaults")
    print("=" * 80)

    # Ensure API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    # Create pipeline with defaults
    pipeline = create_pipeline()

    # Initialize
    pipeline.initialize()

    # Ingest and index documents
    # Replace with your actual file paths
    documents_path = "./sample_documents"  # Directory with your documents

    if not Path(documents_path).exists():
        print(f"⚠️  Directory {documents_path} not found. Creating sample text...")
        # Create sample text document
        sample_text = """
        Artificial Intelligence (AI) is transforming the world. Machine learning,
        a subset of AI, enables computers to learn from data without explicit programming.
        Deep learning uses neural networks with multiple layers to process complex patterns.
        Natural Language Processing (NLP) allows machines to understand and generate human language.
        """
        pipeline.ingest_documents([{"type": "text", "text": sample_text, "metadata": {"source": "sample"}}])
    else:
        # Ingest from directory
        pipeline.run_ingestion(documents_path, source_type="directory")

    # Query the pipeline
    query = "What is machine learning?"
    print(f"\n📝 Query: {query}")

    result = pipeline.answer_question(query, include_sources=True)

    print(f"\n✅ Answer: {result['answer']}")
    print(f"\n📊 Retrieved {result['num_retrieved']} documents")
    print(f"📊 Relevance scores: {result['retrieval_scores']}")

    print("\n" + "=" * 80 + "\n")


def example_2_custom_configuration():
    """
    Example 2: Custom configuration with different providers.

    Demonstrates configuring specific models and parameters.
    """
    print("=" * 80)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 80)

    # Create custom configuration
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-large",
            batch_size=50
        ),
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1500
        ),
        chunking=ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            respect_sentence_boundaries=True
        ),
        vectorstore=VectorStoreConfig(
            provider="chroma",
            collection_name="my_custom_collection",
            persist_directory="./my_vectorstore"
        ),
        retrieval=RetrievalConfig(
            top_k=3,
            search_type="similarity"
        ),
        verbose=True
    )

    # Create pipeline with custom config
    pipeline = RAGPipeline(config)
    pipeline.initialize()

    # Load sample documents
    sample_docs = [
        {
            "type": "text",
            "text": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "type": "text",
            "text": "JavaScript is the programming language of the web, running in browsers and servers.",
            "metadata": {"topic": "programming", "language": "javascript"}
        },
        {
            "type": "text",
            "text": "React is a JavaScript library for building user interfaces, developed by Facebook.",
            "metadata": {"topic": "frameworks", "language": "javascript"}
        }
    ]

    docs = pipeline.ingest_documents(sample_docs)
    pipeline.index_documents(docs)

    # Query
    query = "What is Python?"
    result = pipeline.answer_question(query)

    print(f"\n📝 Query: {query}")
    print(f"✅ Answer: {result['answer']}\n")

    print("=" * 80 + "\n")


def example_3_multiple_sources():
    """
    Example 3: Loading from multiple source types.

    Demonstrates loading from files, URLs, and direct text.
    """
    print("=" * 80)
    print("EXAMPLE 3: Multiple Source Types")
    print("=" * 80)

    pipeline = create_pipeline()
    pipeline.initialize()

    # Mix of different sources
    sources = [
        # Direct text
        {
            "type": "text",
            "text": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
            "metadata": {"source": "manual_entry", "topic": "landmarks"}
        },
        # You can add file paths
        # "path/to/document.pdf",

        # You can add URLs (requires internet)
        # "https://example.com/article",
    ]

    docs = pipeline.ingest_documents(sources)
    pipeline.index_documents(docs)

    # Query
    query = "Where is the Eiffel Tower?"
    answer = pipeline.query(query)  # Simplified method returning just the answer

    print(f"\n📝 Query: {query}")
    print(f"✅ Answer: {answer}\n")

    print("=" * 80 + "\n")


def example_4_using_different_providers():
    """
    Example 4: Using different embedding and LLM providers.

    Shows how to use Anthropic Claude, Google, HuggingFace, etc.
    """
    print("=" * 80)
    print("EXAMPLE 4: Different Providers")
    print("=" * 80)

    # Example with Anthropic Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n--- Using Anthropic Claude ---")
        config = RAGConfig()
        config.llm.provider = "anthropic"
        config.llm.model_name = "claude-3-5-sonnet-20241022"
        config.embedding.provider = "openai"  # Still using OpenAI for embeddings

        pipeline = RAGPipeline(config)
        pipeline.initialize()

        # Add sample data
        sample_text = "Claude is an AI assistant created by Anthropic. It's designed to be helpful, harmless, and honest."
        pipeline.run_ingestion([{"type": "text", "text": sample_text}])

        answer = pipeline.query("Who created Claude?")
        print(f"Answer: {answer}")

    # Example with Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        print("\n--- Using Google Gemini ---")
        config = RAGConfig()
        config.llm.provider = "google"
        config.llm.model_name = "gemini-1.5-pro"
        config.embedding.provider = "google"
        config.embedding.model_name = "text-embedding-004"

        pipeline = RAGPipeline(config)
        pipeline.initialize()

        # Add sample data
        sample_text = "Google Gemini is a family of AI models developed by Google DeepMind."
        pipeline.run_ingestion([{"type": "text", "text": sample_text}])

        answer = pipeline.query("What is Gemini?")
        print(f"Answer: {answer}")

    # Example with local Ollama
    print("\n--- Using Local Ollama (if available) ---")
    try:
        config = RAGConfig()
        config.llm.provider = "ollama"
        config.llm.model_name = "llama2"
        config.embedding.provider = "ollama"
        config.embedding.model_name = "nomic-embed-text"

        pipeline = RAGPipeline(config)
        pipeline.initialize()

        sample_text = "Ollama allows you to run large language models locally on your machine."
        pipeline.run_ingestion([{"type": "text", "text": sample_text}])

        answer = pipeline.query("What is Ollama?")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Ollama not available: {e}")

    print("\n" + "=" * 80 + "\n")


def example_5_advanced_retrieval():
    """
    Example 5: Advanced retrieval with reranking.

    Demonstrates using reranking for better results.
    """
    print("=" * 80)
    print("EXAMPLE 5: Advanced Retrieval with Reranking")
    print("=" * 80)

    config = RAGConfig()
    config.retrieval.enable_reranking = True
    config.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    config.retrieval.top_k = 10
    config.retrieval.reranker_top_k = 3

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    # Add multiple documents
    docs = [
        {"type": "text", "text": f"Document {i}: This is sample content about topic {i % 3}."}
        for i in range(20)
    ]
    pipeline.run_ingestion(docs)

    # Query with reranking
    query = "Tell me about topic 1"
    result = pipeline.answer_question(query, enable_reranking=True, return_raw_results=True)

    print(f"\n📝 Query: {query}")
    print(f"✅ Answer: {result['answer']}")
    print(f"\n📊 Initially retrieved: 10 documents")
    print(f"📊 After reranking: {len(result['source_documents'])} documents")

    print("\n" + "=" * 80 + "\n")


def example_6_persistence():
    """
    Example 6: Persistence and reloading.

    Shows how to save and reload the vector store.
    """
    print("=" * 80)
    print("EXAMPLE 6: Persistence")
    print("=" * 80)

    persist_dir = "./persistent_vectorstore"

    # First run: Create and save
    print("\n--- Creating and persisting vector store ---")
    config = RAGConfig()
    config.vectorstore.persist_directory = persist_dir
    config.vectorstore.collection_name = "persistent_collection"

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    sample_text = "This data will be persisted across sessions."
    pipeline.run_ingestion([{"type": "text", "text": sample_text}])
    pipeline.save()

    print("✅ Data saved to disk")

    # Second run: Reload from disk
    print("\n--- Reloading from persisted vector store ---")
    pipeline2 = RAGPipeline(config)
    pipeline2.initialize()

    # The data should already be there
    answer = pipeline2.query("What will be persisted?")
    print(f"Answer from reloaded pipeline: {answer}")

    print("\n" + "=" * 80 + "\n")


def interactive_mode():
    """
    Interactive mode for querying the pipeline.
    """
    print("=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nInitializing pipeline...")

    pipeline = create_pipeline()
    pipeline.initialize()

    # Load sample data
    print("\nLoading sample data...")
    sample_data = """
    The Universal RAG Pipeline is a production-ready system for Retrieval-Augmented Generation.
    It supports multiple embedding providers including OpenAI, Google, HuggingFace, and Cohere.
    The pipeline can work with various LLM providers like GPT-4, Claude, Gemini, and local models.
    It includes features like document chunking, vector storage, semantic search, and answer generation.
    """

    pipeline.run_ingestion([{"type": "text", "text": sample_data}])

    print("\n✅ Pipeline ready! Type your questions (or 'quit' to exit)")
    print("-" * 80)

    while True:
        try:
            query = input("\n🤔 Your question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not query:
                continue

            result = pipeline.answer_question(query, include_sources=False)
            print(f"\n✅ Answer: {result['answer']}")
            print(f"   (Retrieved from {result['num_retrieved']} sources)")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main function to run examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "UNIVERSAL RAG PIPELINE DEMO" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "interactive":
            interactive_mode()
        elif mode == "1":
            example_1_quick_start()
        elif mode == "2":
            example_2_custom_configuration()
        elif mode == "3":
            example_3_multiple_sources()
        elif mode == "4":
            example_4_using_different_providers()
        elif mode == "5":
            example_5_advanced_retrieval()
        elif mode == "6":
            example_6_persistence()
        else:
            print(f"Unknown mode: {mode}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage information."""
    print("Usage: python run_query.py [mode]")
    print("\nAvailable modes:")
    print("  1          - Quick start example")
    print("  2          - Custom configuration example")
    print("  3          - Multiple sources example")
    print("  4          - Different providers example")
    print("  5          - Advanced retrieval example")
    print("  6          - Persistence example")
    print("  interactive - Interactive query mode")
    print("\nExample:")
    print("  python run_query.py interactive")
    print("  python run_query.py 1")
    print("\n")


if __name__ == "__main__":
    main()
