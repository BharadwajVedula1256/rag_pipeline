"""
Example script showing how to use the RAG pipeline with .env file.

This demonstrates loading API keys from a .env file instead of
hardcoding them or setting them manually.
"""

from utils import load_env_file, print_env_status, validate_provider_setup
from pipeline import create_pipeline


def main():
    """Main example function."""
    print("\n" + "=" * 70)
    print("RAG Pipeline - Using .env File for Configuration")
    print("=" * 70 + "\n")

    # Step 1: Load environment variables from .env file
    print("Step 1: Loading .env file...")
    env_loaded = load_env_file()

    if env_loaded:
        print("✓ .env file loaded successfully\n")
    else:
        print("⚠️  .env file not found or python-dotenv not installed")
        print("   Copy .env.example to .env and fill in your API keys\n")
        return

    # Step 2: Check environment status
    print("Step 2: Checking environment variables...\n")
    print_env_status()
    print()

    # Step 3: Validate that required keys are set
    print("Step 3: Validating provider setup...")
    embedding_provider = "openai"  # Change as needed
    llm_provider = "openai"  # Change as needed

    is_valid = validate_provider_setup(embedding_provider, llm_provider)

    if not is_valid:
        print("\n❌ Required API keys not set. Please update your .env file.")
        return

    print("✓ All required API keys are set\n")

    # Step 4: Create and initialize pipeline
    print("Step 4: Creating RAG pipeline...")
    print(f"   Embedding Provider: {embedding_provider}")
    print(f"   LLM Provider: {llm_provider}\n")

    pipeline = create_pipeline(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider
    )

    pipeline.initialize()
    print("✓ Pipeline initialized successfully\n")

    # Step 5: Load sample data
    print("Step 5: Loading sample data...")
    sample_data = [
        {
            "type": "text",
            "text": """
                The Universal RAG Pipeline is a production-ready system for
                Retrieval-Augmented Generation. It supports multiple providers
                including OpenAI, Anthropic Claude, Google Gemini, and more.
                Configuration can be done via environment variables, .env files,
                or programmatically in code.
            """,
            "metadata": {"source": "documentation", "topic": "rag_pipeline"}
        }
    ]

    pipeline.run_ingestion(sample_data)
    print("✓ Sample data indexed\n")

    # Step 6: Query the pipeline
    print("Step 6: Querying the pipeline...\n")

    questions = [
        "What is the Universal RAG Pipeline?",
        "Which providers are supported?",
        "How can I configure the pipeline?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        answer = pipeline.query(question)
        print(f"Answer: {answer}\n")

    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70 + "\n")


def quick_example():
    """
    Quick example with minimal setup.

    This assumes you already have a .env file with your API keys.
    """
    from utils import load_env_file
    from pipeline import create_pipeline

    # Load .env file
    load_env_file()

    # Create pipeline (will auto-detect API keys from environment)
    pipeline = create_pipeline()
    pipeline.initialize()

    # Add documents
    pipeline.run_ingestion([
        {"type": "text", "text": "Your document content here"}
    ])

    # Query
    answer = pipeline.query("Your question here")
    print(answer)


def example_with_custom_providers():
    """Example using different embedding and LLM providers."""
    from utils import load_env_file
    from pipeline import create_pipeline

    # Load .env file
    load_env_file()

    # Use Google for embeddings and Anthropic for LLM
    pipeline = create_pipeline(
        embedding_provider="google",
        embedding_model="text-embedding-004",
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022"
    )

    pipeline.initialize()

    # Rest of your code...
    print("Pipeline created with Google embeddings and Anthropic Claude LLM")


def example_check_api_keys():
    """Example to check which API keys are set."""
    from utils import load_env_file, check_required_keys

    load_env_file()

    # Check specific providers
    providers = ["openai", "anthropic", "google", "cohere"]
    status = check_required_keys(providers)

    print("API Key Status:")
    for provider, is_set in status.items():
        status_icon = "✓" if is_set else "✗"
        print(f"  {status_icon} {provider.upper()}")


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run other examples:
    # quick_example()
    # example_with_custom_providers()
    # example_check_api_keys()
