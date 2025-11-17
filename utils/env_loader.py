"""
Environment variable loader utility.

Provides helper functions to load configuration from .env files.
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file (defaults to .env in current directory)

    Returns:
        True if .env file was loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        return False

    if env_path is None:
        env_path = Path.cwd() / ".env"

    if Path(env_path).exists():
        load_dotenv(env_path)
        return True
    else:
        print(f"Warning: .env file not found at {env_path}")
        return False


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider.

    Args:
        provider: Provider name (openai, anthropic, google, cohere, huggingface)

    Returns:
        API key string or None if not found
    """
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "pinecone": "PINECONE_API_KEY",
        "qdrant": "QDRANT_API_KEY",
        "weaviate": "WEAVIATE_API_KEY"
    }

    env_var = env_var_map.get(provider.lower())
    if env_var:
        return os.getenv(env_var)
    return None


def check_required_keys(providers: list) -> dict:
    """
    Check if required API keys are set for given providers.

    Args:
        providers: List of provider names

    Returns:
        Dictionary with provider: bool indicating if key is set
    """
    results = {}
    for provider in providers:
        key = get_api_key(provider)
        results[provider] = key is not None and len(key) > 0
    return results


def print_env_status():
    """Print status of all possible API keys."""
    providers = ["openai", "anthropic", "google", "cohere", "huggingface"]

    print("=" * 60)
    print("Environment Variables Status")
    print("=" * 60)

    for provider in providers:
        key = get_api_key(provider)
        status = "✓ SET" if key else "✗ NOT SET"
        masked_key = f"{key[:8]}...{key[-4:]}" if key and len(key) > 12 else "None"
        print(f"{provider.upper():15} {status:10} {masked_key if key else ''}")

    print("=" * 60)


def validate_provider_setup(embedding_provider: str, llm_provider: str) -> bool:
    """
    Validate that required API keys are set for chosen providers.

    Args:
        embedding_provider: Embedding provider name
        llm_provider: LLM provider name

    Returns:
        True if all required keys are set, False otherwise
    """
    # Local providers don't need API keys
    local_providers = ["ollama", "huggingface"]  # HuggingFace can be local

    required_providers = []

    if embedding_provider not in local_providers:
        required_providers.append(embedding_provider)

    if llm_provider not in local_providers:
        required_providers.append(llm_provider)

    if not required_providers:
        return True  # All local, no keys needed

    results = check_required_keys(required_providers)

    all_set = all(results.values())

    if not all_set:
        print("⚠️  Missing required API keys:")
        for provider, is_set in results.items():
            if not is_set:
                print(f"   - {provider.upper()}_API_KEY not set")
        print("\nPlease set the required environment variables or use a .env file")

    return all_set
