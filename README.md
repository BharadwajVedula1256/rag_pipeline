# Universal RAG Pipeline

A complete, production-ready, modular Retrieval-Augmented Generation (RAG) pipeline in Python. Use it anywhere - with CrewAI, LangChain, LlamaIndex, FastAPI, CLI, or standalone scripts.

## Features

- **🔌 Universal Compatibility**: Works with any embedding model or LLM
- **📚 Multi-Format Support**: PDFs, text files, markdown, HTML, DOCX, CSV, JSON
- **🧩 Modular Architecture**: Clean, dependency-injected components
- **🎯 Multiple Providers**:
  - **Embeddings**: OpenAI, Google, HuggingFace, Cohere, Ollama
  - **LLMs**: OpenAI (GPT-4+), Anthropic (Claude), Google (Gemini), Cohere, Ollama
  - **Vector Stores**: ChromaDB, FAISS (Pinecone, Qdrant, Weaviate support ready)
- **⚡ Advanced Features**: Reranking, MMR search, filtering, persistence
- **🔧 Fully Configurable**: JSON configs, environment variables, programmatic setup

## Installation

```bash
# Clone or download the repository
cd rag_pipeline

# Install dependencies
pip install -r requirements.txt
```

### Minimal Installation

For a minimal setup (OpenAI + ChromaDB):

```bash
pip install openai chromadb tiktoken beautifulsoup4 pypdf numpy requests
```

## Quick Start

### 1. Set API Keys

```bash
export OPENAI_API_KEY="your-key-here"
# Or for other providers:
# export ANTHROPIC_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"
# export COHERE_API_KEY="your-key-here"
```

### 2. Basic Usage

```python
from pipeline import create_pipeline

# Create pipeline with defaults (OpenAI + ChromaDB)
pipeline = create_pipeline()

# Initialize all components
pipeline.initialize()

# Ingest and index documents
pipeline.run_ingestion("./documents")  # Directory
# or
pipeline.run_ingestion("document.pdf")  # Single file
# or
pipeline.run_ingestion("https://example.com/article")  # URL

# Query
answer = pipeline.query("What is this document about?")
print(answer)
```

### 3. Run Demo

```bash
# Interactive mode
python run_query.py interactive

# Run specific examples
python run_query.py 1  # Quick start
python run_query.py 2  # Custom configuration
python run_query.py 3  # Multiple sources
python run_query.py 4  # Different providers
python run_query.py 5  # Advanced retrieval
python run_query.py 6  # Persistence
```

## Architecture

```
rag_pipeline/
├── ingest/          # Document loading and cleaning
│   ├── loader.py    # Load files, URLs, directories
│   ├── extractor.py # Extract text from various formats
│   └── cleaner.py   # Text normalization
├── chunking/        # Text chunking strategies
│   └── chunker.py   # Token-based, sentence-based chunking
├── embeddings/      # Embedding generation
│   └── embedder.py  # Multi-provider embedding support
├── vectorstore/     # Vector storage
│   └── store.py     # ChromaDB, FAISS, etc.
├── retrieval/       # Document retrieval
│   └── retriever.py # Similarity search, MMR, reranking
├── llm/             # LLM answer generation
│   └── generator.py # Multi-provider LLM support
├── utils/           # Utilities
│   ├── config.py    # Configuration management
│   └── logger.py    # Logging utilities
├── pipeline.py      # Main pipeline orchestration
└── run_query.py     # Demo script
```

## Configuration

### Programmatic Configuration

```python
from pipeline import RAGPipeline
from utils import RAGConfig, EmbeddingConfig, LLMConfig

config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-large"
    ),
    llm=LLMConfig(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.7
    ),
    chunking=ChunkingConfig(
        chunk_size=800,
        chunk_overlap=100
    ),
    retrieval=RetrievalConfig(
        top_k=5,
        enable_reranking=True
    )
)

pipeline = RAGPipeline(config)
```

### JSON Configuration

```python
# Save config
config.save_to_json("my_config.json")

# Load from file
pipeline = RAGPipeline.from_config_file("my_config.json")
```

## Usage Examples

### Example 1: Different Embedding Providers

```python
# OpenAI
config.embedding.provider = "openai"
config.embedding.model_name = "text-embedding-3-large"

# Google
config.embedding.provider = "google"
config.embedding.model_name = "text-embedding-004"

# HuggingFace (local)
config.embedding.provider = "huggingface"
config.embedding.model_name = "BAAI/bge-large-en-v1.5"

# Cohere
config.embedding.provider = "cohere"
config.embedding.model_name = "embed-english-v3.0"

# Ollama (local)
config.embedding.provider = "ollama"
config.embedding.model_name = "nomic-embed-text"
```

### Example 2: Different LLM Providers

```python
# OpenAI GPT-4
config.llm.provider = "openai"
config.llm.model_name = "gpt-4-turbo"

# Anthropic Claude
config.llm.provider = "anthropic"
config.llm.model_name = "claude-3-5-sonnet-20241022"

# Google Gemini
config.llm.provider = "google"
config.llm.model_name = "gemini-1.5-pro"

# Cohere
config.llm.provider = "cohere"
config.llm.model_name = "command-r-plus"

# Ollama (local)
config.llm.provider = "ollama"
config.llm.model_name = "llama2"
```

### Example 3: Custom Providers

```python
from embeddings import CustomEmbedding, Embedder
from llm import CustomLLM, Generator

# Custom embedding function
def my_embed_function(text: str) -> list:
    # Your custom embedding logic
    return [0.1, 0.2, 0.3, ...]  # Return embedding vector

custom_embedder = Embedder(
    CustomEmbedding(
        embed_func=my_embed_function,
        dimension=1536
    )
)

# Custom LLM function
def my_generate_function(prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
    # Your custom LLM logic
    return "Generated response"

custom_generator = Generator(
    CustomLLM(generate_func=my_generate_function)
)
```

### Example 4: Advanced Retrieval

```python
from retrieval import MMRSearchStrategy

# Use MMR (Maximal Marginal Relevance) for diversity
config.retrieval.search_type = "mmr"

# Enable reranking for better accuracy
config.retrieval.enable_reranking = True
config.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Query with custom parameters
result = pipeline.answer_question(
    query="What is AI?",
    top_k=10,
    enable_reranking=True,
    include_sources=True
)
```

### Example 5: Multiple Document Sources

```python
# Mix of files, URLs, and direct text
sources = [
    "document.pdf",
    "https://example.com/article",
    {"type": "text", "text": "Direct text content", "metadata": {"source": "manual"}},
    "./documents_directory"
]

pipeline.run_ingestion(sources)
```

### Example 6: Integration with Other Frameworks

```python
# Use with CrewAI
from crewai import Agent, Task

def rag_search(query: str) -> str:
    return pipeline.query(query)

agent = Agent(
    role="Research Assistant",
    tools=[rag_search],
    # ... other config
)

# Use with LangChain
from langchain.tools import Tool

rag_tool = Tool(
    name="RAG Search",
    func=pipeline.query,
    description="Search the knowledge base"
)

# Use with FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.post("/query")
async def query_rag(query: str):
    result = pipeline.answer_question(query)
    return result
```

## Advanced Features

### Persistence

```python
# Vector store automatically persists with ChromaDB
pipeline.save()  # Explicit save

# Reload in a new session
pipeline2 = RAGPipeline(config)
pipeline2.initialize()  # Automatically loads persisted data
```

### Custom Chunking

```python
from chunking import CustomChunker

def my_chunking_function(text: str) -> list:
    # Custom chunking logic
    return text.split("\n\n")  # Example: split by paragraphs

chunker = DocumentChunker(
    strategy=CustomChunker(my_chunking_function)
)
```

### Metadata Filtering

```python
# Add documents with metadata
docs = pipeline.ingest_documents([
    {"type": "text", "text": "Content 1", "metadata": {"category": "tech"}},
    {"type": "text", "text": "Content 2", "metadata": {"category": "science"}}
])
pipeline.index_documents(docs)

# Filter during retrieval
results = pipeline.retriever.retrieve(
    query="What is AI?",
    k=5,
    filter_dict={"category": "tech"}
)
```

## API Reference

### RAGPipeline

Main pipeline class.

**Methods:**
- `initialize()` - Initialize all components
- `ingest_documents(sources)` - Load documents
- `index_documents(documents)` - Index into vector store
- `run_ingestion(sources)` - Combined load + index
- `answer_question(query, **kwargs)` - Get detailed answer
- `query(query)` - Get simple answer string
- `clear_index()` - Clear vector store
- `save()` - Persist to disk

### Configuration Classes

- `RAGConfig` - Main configuration
- `EmbeddingConfig` - Embedding settings
- `LLMConfig` - LLM settings
- `ChunkingConfig` - Chunking settings
- `VectorStoreConfig` - Vector store settings
- `RetrievalConfig` - Retrieval settings

## Supported Formats

- **Documents**: PDF, DOCX, TXT, MD, HTML, CSV, JSON
- **Sources**: Local files, directories, URLs, direct text
- **Embeddings**: OpenAI, Google, HuggingFace, Cohere, Ollama, Custom
- **LLMs**: OpenAI, Anthropic, Google, Cohere, Ollama, Custom
- **Vector Stores**: ChromaDB, FAISS (+ extensible for others)

## Environment Variables

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
HUGGINGFACE_API_KEY=...

# Optional: Custom endpoints
OLLAMA_BASE_URL=http://localhost:11434
```

## Performance Tips

1. **Batch Processing**: Use appropriate batch sizes for embeddings
2. **Chunk Size**: Balance between context (larger) and precision (smaller)
3. **Reranking**: Use for better accuracy, but adds latency
4. **Local Models**: Use Ollama for privacy and no API costs
5. **Vector Store**: FAISS for speed, ChromaDB for persistence

## Troubleshooting

### Import Errors

```bash
# Install missing dependencies
pip install <missing-package>
```

### API Key Errors

```python
# Check if API key is set
import os
print(os.getenv("OPENAI_API_KEY"))
```

### Vector Store Issues

```python
# Clear and rebuild
pipeline.clear_index()
pipeline.run_ingestion(sources)
```

## Contributing

This is a modular system designed for extension:

1. Add new extractors in `ingest/extractor.py`
2. Add new embeddings in `embeddings/embedder.py`
3. Add new LLMs in `llm/generator.py`
4. Add new vector stores in `vectorstore/store.py`

All components use abstract base classes for easy extension.

## License

MIT License - Use freely in commercial and personal projects.

## Support

For issues and questions, check the code comments and docstrings - they're comprehensive!

---

**Built with ❤️ for the AI community**
