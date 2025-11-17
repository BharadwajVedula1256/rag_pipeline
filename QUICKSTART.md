# Quick Start Guide

Get started with the Universal RAG Pipeline in 5 minutes!

## Step 1: Install

```bash
pip install -r requirements.txt
```

## Step 2: Set API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Step 3: Create Your First RAG Application

Create a file `my_rag_app.py`:

```python
from pipeline import create_pipeline

# Create and initialize pipeline
pipeline = create_pipeline()
pipeline.initialize()

# Add your documents
pipeline.run_ingestion([
    {"type": "text", "text": """
        Python is a high-level programming language.
        It was created by Guido van Rossum in 1991.
        Python emphasizes code readability and simplicity.
    """}
])

# Ask questions
answer = pipeline.query("Who created Python?")
print(f"Answer: {answer}")
```

## Step 4: Run It

```bash
python my_rag_app.py
```

That's it! You now have a working RAG system.

## Next Steps

### Add Real Documents

```python
# From a file
pipeline.run_ingestion("document.pdf")

# From a directory
pipeline.run_ingestion("./my_documents")

# From a URL
pipeline.run_ingestion("https://example.com/article")
```

### Use Different Providers

```python
from pipeline import RAGPipeline
from utils import RAGConfig, EmbeddingConfig, LLMConfig

config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="google",  # or "huggingface", "cohere", "ollama"
        model_name="text-embedding-004"
    ),
    llm=LLMConfig(
        provider="anthropic",  # or "google", "cohere", "ollama"
        model_name="claude-3-5-sonnet-20241022"
    )
)

pipeline = RAGPipeline(config)
```

### Try Interactive Mode

```bash
python run_query.py interactive
```

### See More Examples

```bash
python run_query.py 1  # Quick start
python run_query.py 2  # Custom configuration
python run_query.py 3  # Multiple sources
python run_query.py 4  # Different providers
python run_query.py 5  # Advanced retrieval
python run_query.py 6  # Persistence
```

## Common Use Cases

### Build a Document Q&A System

```python
pipeline = create_pipeline()
pipeline.initialize()
pipeline.run_ingestion("./company_docs")

while True:
    question = input("Ask a question: ")
    answer = pipeline.query(question)
    print(f"Answer: {answer}\n")
```

### Integrate with FastAPI

```python
from fastapi import FastAPI
from pipeline import create_pipeline

app = FastAPI()
pipeline = create_pipeline()
pipeline.initialize()
pipeline.run_ingestion("./knowledge_base")

@app.post("/query")
async def query(question: str):
    return {"answer": pipeline.query(question)}
```

### Use with LangChain

```python
from langchain.tools import Tool
from pipeline import create_pipeline

pipeline = create_pipeline()
pipeline.initialize()
pipeline.run_ingestion("./docs")

rag_tool = Tool(
    name="Knowledge Base",
    func=pipeline.query,
    description="Search the knowledge base for information"
)
```

## Troubleshooting

**Problem**: Import errors
```bash
pip install -r requirements.txt
```

**Problem**: API key not found
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

**Problem**: Out of memory
```python
# Reduce chunk size and batch size
config.chunking.chunk_size = 400
config.embedding.batch_size = 50
```

## Next: Read the Full README

See [README.md](README.md) for comprehensive documentation and advanced features.
