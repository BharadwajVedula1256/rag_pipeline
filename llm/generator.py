"""
LLM generation for answering queries using retrieved context.

Supports OpenAI, Anthropic, Google, Cohere, Ollama, and custom LLM providers.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate answer using query and context.

        Args:
            query: User query
            context: List of context strings
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated answer
        """
        pass


class OpenAILLM(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate answer using query and context."""
        # Build context string
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])

        # Build messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say so clearly."
            )
            messages.append({"role": "system", "content": default_system})

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )

        return response.choices[0].message.content


class AnthropicLLM(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize Anthropic LLM.

        Args:
            api_key: Anthropic API key
            model: Model name (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.)
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate answer using query and context."""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])

        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say so clearly."
            )

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        return response.content[0].text


class GoogleLLM(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize Google LLM.

        Args:
            api_key: Google API key
            model: Model name (gemini-pro, gemini-1.5-pro, etc.)
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai
            self.model_instance = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        generation_config = {
            "temperature": temperature or self.temperature,
            "max_output_tokens": max_tokens or self.max_tokens,
        }

        response = self.model_instance.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate answer using query and context."""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])

        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say so clearly."
            )

        prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        return self.generate(prompt, **kwargs)


class CohereLLM(LLMProvider):
    """Cohere LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-plus",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize Cohere LLM.

        Args:
            api_key: Cohere API key
            model: Model name (command, command-light, command-r, command-r-plus)
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            import cohere
            self.client = cohere.Client(api_key)
        except ImportError:
            raise ImportError("cohere package required. Install with: pip install cohere")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        response = self.client.chat(
            message=prompt,
            model=self.model,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        return response.text

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate answer using query and context."""
        # Cohere has built-in RAG support
        documents = [{"text": ctx} for ctx in context]

        response = self.client.chat(
            message=query,
            model=self.model,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            documents=documents,
            preamble=system_prompt
        )

        return response.text


class OllamaLLM(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7
    ):
        """
        Initialize Ollama LLM.

        Args:
            model: Model name (llama2, mistral, codellama, etc.)
            base_url: Ollama server URL
            temperature: Default temperature
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("ollama package required. Install with: pip install ollama")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        options = {
            "temperature": temperature or self.temperature
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options=options
        )
        return response['response']

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate answer using query and context."""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])

        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say so clearly."
            )

        prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        return self.generate(prompt, **kwargs)


class NebiusLLM(LLMProvider):
    """Nebius AI Studio LLM provider (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url: str = "https://api.studio.nebius.ai/v1/",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize Nebius LLM.

        Args:
            api_key: Nebius API key
            model: Model name (meta-llama/Meta-Llama-3.1-70B-Instruct, Qwen/Qwen2.5-72B-Instruct, etc.)
            base_url: Nebius API base URL
            temperature: Default temperature
            max_tokens: Default max tokens
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate answer using query and context."""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say so clearly."
            )
            messages.append({"role": "system", "content": default_system})

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )

        return response.choices[0].message.content


class CustomLLM(LLMProvider):
    """Custom LLM provider using provided functions."""

    def __init__(
        self,
        generate_func: Callable,
        generate_with_context_func: Optional[Callable] = None
    ):
        """
        Initialize custom LLM.

        Args:
            generate_func: Function for basic generation
            generate_with_context_func: Optional function for context-based generation
        """
        self.generate_func = generate_func
        self.generate_with_context_func = generate_with_context_func

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate using custom function."""
        return self.generate_func(prompt, temperature, max_tokens, **kwargs)

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate with context using custom function."""
        if self.generate_with_context_func:
            return self.generate_with_context_func(query, context, system_prompt, **kwargs)
        else:
            # Fallback to basic generation with formatted prompt
            context_str = "\n\n".join(context)
            prompt = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
            return self.generate(prompt, **kwargs)


class Generator:
    """
    Main generator class for LLM-based answer generation.

    Handles prompt formatting and response generation.
    """

    def __init__(
        self,
        provider: LLMProvider,
        default_system_prompt: Optional[str] = None
    ):
        """
        Initialize generator.

        Args:
            provider: LLMProvider instance
            default_system_prompt: Default system prompt to use
        """
        self.provider = provider
        self.default_system_prompt = default_system_prompt

    def generate(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        include_sources: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer from query and context.

        Args:
            query: User query
            context: List of context strings
            system_prompt: Optional system prompt (overrides default)
            include_sources: Whether to include source information
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with answer and metadata
        """
        system_prompt = system_prompt or self.default_system_prompt

        answer = self.provider.generate_with_context(
            query=query,
            context=context,
            system_prompt=system_prompt,
            **kwargs
        )

        result = {
            "answer": answer,
            "query": query,
            "num_sources": len(context)
        }

        if include_sources:
            result["sources"] = context

        return result

    def generate_simple(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a simple prompt.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Generated text
        """
        return self.provider.generate(prompt, **kwargs)

    @staticmethod
    def create_from_config(config) -> "Generator":
        """
        Create generator from configuration.

        Args:
            config: LLMConfig instance

        Returns:
            Configured Generator instance
        """
        provider_map = {
            "openai": lambda: OpenAILLM(
                api_key=config.api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            ),
            "anthropic": lambda: AnthropicLLM(
                api_key=config.api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            ),
            "google": lambda: GoogleLLM(
                api_key=config.api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            ),
            "cohere": lambda: CohereLLM(
                api_key=config.api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            ),
            "ollama": lambda: OllamaLLM(
                model=config.model_name,
                base_url=config.custom_endpoint or "http://localhost:11434",
                temperature=config.temperature
            ),
            "nebius": lambda: NebiusLLM(
                api_key=config.api_key,
                model=config.model_name,
                base_url=config.custom_endpoint or "https://api.studio.nebius.ai/v1/",
                temperature=config.temperature,
                max_tokens=config.max_tokens
            ),
        }

        if config.provider not in provider_map:
            raise ValueError(f"Unknown LLM provider: {config.provider}")

        provider = provider_map[config.provider]()
        return Generator(provider, config.system_prompt)
