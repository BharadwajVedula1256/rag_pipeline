"""
Example script showing how to use Nebius AI Studio with the RAG pipeline.

Nebius AI Studio provides both embeddings and LLMs with an OpenAI-compatible API.
This example demonstrates how to use Nebius for both components.
"""

from utils import load_env_file, RAGConfig, EmbeddingConfig, LLMConfig
from pipeline import RAGPipeline


def example_nebius_embeddings_only():
    """
    Example using Nebius for embeddings only.

    Uses Nebius BAAI/bge-en-icl embedding model with OpenAI GPT-4 for LLM.
    """
    print("\n" + "=" * 70)
    print("Example 1: Nebius Embeddings + OpenAI LLM")
    print("=" * 70 + "\n")

    # Load environment variables
    load_env_file()

    # Configure pipeline with Nebius embeddings
    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="nebius",
            model_name="BAAI/bge-en-icl",  # Nebius embedding model
            batch_size=100
        ),
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-4"
        )
    )

    # Create and initialize pipeline
    pipeline = RAGPipeline(config)
    pipeline.initialize()

    print("✓ Pipeline initialized with Nebius embeddings and OpenAI LLM\n")

    # Add sample data
    sample_text = """
    Nebius AI Studio is a cloud platform providing enterprise-grade AI infrastructure.
    It offers OpenAI-compatible APIs for both embeddings and language models.
    Nebius supports models like Meta-Llama, Qwen, and BAAI embeddings.
    """

    pipeline.run_ingestion([{"type": "text", "text": sample_text}])
    print("✓ Sample data indexed\n")

    # Query
    query = "What is Nebius AI Studio?"
    answer = pipeline.query(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}\n")


def example_nebius_llm_only():
    """
    Example using Nebius for LLM only.

    Uses OpenAI embeddings with Nebius Meta-Llama LLM.
    """
    print("\n" + "=" * 70)
    print("Example 2: OpenAI Embeddings + Nebius LLM")
    print("=" * 70 + "\n")

    load_env_file()

    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-large"
        ),
        llm=LLMConfig(
            provider="nebius",
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",  # Nebius LLM model
            temperature=0.7,
            max_tokens=2000
        )
    )

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    print("✓ Pipeline initialized with OpenAI embeddings and Nebius LLM\n")

    sample_text = """
    Meta-Llama is a family of large language models developed by Meta AI.
    Llama 3.1 is available in sizes ranging from 8B to 70B parameters.
    These models excel at instruction following and reasoning tasks.
    """

    pipeline.run_ingestion([{"type": "text", "text": sample_text}])
    print("✓ Sample data indexed\n")

    query = "What is Meta-Llama?"
    answer = pipeline.query(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}\n")


def example_full_nebius_stack():
    """
    Example using Nebius for both embeddings and LLM.

    Full Nebius stack for maximum integration.
    """
    print("\n" + "=" * 70)
    print("Example 3: Full Nebius Stack (Embeddings + LLM)")
    print("=" * 70 + "\n")

    load_env_file()

    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="nebius",
            model_name="BAAI/bge-en-icl",
            batch_size=100
        ),
        llm=LLMConfig(
            provider="nebius",
            model_name="Qwen/Qwen2.5-72B-Instruct",  # Alternative Nebius LLM
            temperature=0.7,
            max_tokens=1500
        )
    )

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    print("✓ Pipeline initialized with full Nebius stack\n")

    sample_docs = [
        {
            "type": "text",
            "text": "Qwen is a series of large language models developed by Alibaba Cloud.",
            "metadata": {"topic": "qwen"}
        },
        {
            "type": "text",
            "text": "Nebius provides infrastructure for deploying AI models at scale.",
            "metadata": {"topic": "nebius"}
        }
    ]

    pipeline.run_ingestion(sample_docs)
    print("✓ Sample data indexed\n")

    queries = [
        "What is Qwen?",
        "What does Nebius provide?"
    ]

    for query in queries:
        answer = pipeline.query(query)
        print(f"Query: {query}")
        print(f"Answer: {answer}\n")


def example_nebius_with_custom_endpoint():
    """
    Example using custom Nebius endpoint.

    Shows how to configure custom API endpoints if needed.
    """
    print("\n" + "=" * 70)
    print("Example 4: Nebius with Custom Endpoint")
    print("=" * 70 + "\n")

    load_env_file()

    # Custom endpoint configuration
    custom_endpoint = "https://api.studio.nebius.ai/v1/"  # Default endpoint

    config = RAGConfig(
        embedding=EmbeddingConfig(
            provider="nebius",
            model_name="BAAI/bge-en-icl",
            custom_endpoint=custom_endpoint
        ),
        llm=LLMConfig(
            provider="nebius",
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
            custom_endpoint=custom_endpoint,
            system_prompt="You are a helpful AI assistant specialized in technical documentation."
        )
    )

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    print(f"✓ Pipeline initialized with custom endpoint: {custom_endpoint}\n")

    sample_text = """
    Cloud-native AI deployment requires scalable infrastructure.
    Nebius provides enterprise-grade solutions for AI workloads.
    """

    pipeline.run_ingestion([{"type": "text", "text": sample_text}])
    print("✓ Sample data indexed\n")

    query = "What is needed for cloud-native AI deployment?"
    answer = pipeline.query(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}\n")


def main():
    """Run all Nebius examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "NEBIUS AI STUDIO RAG EXAMPLES" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")

    # Check if Nebius API key is set
    import os
    if not os.getenv("NEBIUS_API_KEY"):
        print("\n⚠️  Warning: NEBIUS_API_KEY not set in environment")
        print("   Please set it in your .env file or environment:")
        print("   export NEBIUS_API_KEY='your-nebius-api-key'\n")
        print("   Visit https://studio.nebius.ai to get your API key\n")
        return

    try:
        # Run examples
        example_nebius_embeddings_only()
        example_nebius_llm_only()
        example_full_nebius_stack()
        example_nebius_with_custom_endpoint()

        print("=" * 70)
        print("✅ All Nebius examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("Make sure:")
        print("1. NEBIUS_API_KEY is set in your .env file")
        print("2. You have network connectivity")
        print("3. The Nebius API is accessible\n")


if __name__ == "__main__":
    main()
