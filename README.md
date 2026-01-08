# RAG-EPO

A Retrieval Augmented Generation (RAG) system for querying documentation. This project uses LangChain, ChromaDB, and OpenAI to create an intelligent question-answering system that can retrieve and answer questions about domain-specific documentation.

## Features

- **Document Ingestion**: Process and chunk documents into a vector database
- **Semantic Search**: Retrieve relevant context using OpenAI embeddings
- **Question Answering**: Generate accurate answers using GPT-4.1-nano with retrieved context
- **Conversational Interface**: Chat interface built with Gradio (currently configured but commented out)
- **Evaluation Framework**: Comprehensive evaluation tools to assess retrieval and answer quality

## Requirements

- Python >= 3.12
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_epo
```

2. Install dependencies:
```bash
pip install -e .
```

Or if using `uv`:
```bash
uv sync
```

3. Set up environment variables:
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
rag_epo/
├── data/                  # Document files
├── src/
│   ├── ingest.py         # Document ingestion and vector database creation
│   ├── retrieve.py       # RAG retrieval and question answering
│   ├── evaluator.py      # Gradio dashboard for evaluation visualization
│   └── evaluation/       # Evaluation framework
│       ├── eval.py       # Core evaluation functions and CLI interface
│       ├── test.py       # Test loading utilities and data models
│       └── tests.jsonl   # Test cases in JSONL format
├── vector_db/            # ChromaDB vector database (created after ingestion)
├── main.py               # Main entry point
└── pyproject.toml        # Project dependencies
```

## Usage

### 1. Ingest Documents

Process your documents and create the vector database:

```bash
python src/ingest.py
```

This script will:
- Load all `.txt` files from the `data/` directory
- Split documents into chunks (1500 characters with 200 character overlap)
- Create embeddings using OpenAI's `text-embedding-3-large` model
- Store everything in a ChromaDB vector database at `vector_db/`

### 2. Query the System

To use the retrieval system, you can either:

**Option A: Use the Gradio Interface** :

Run:
```bash
python src/app.py
```

**Option B: Use the functions programmatically**:
```python
from src.retrieve import answer_question

question = "What is the main purpose of this document?"
answer, docs = answer_question(question)
print(answer)
```

### 3. Evaluate the System

The evaluation framework provides comprehensive tools to assess both retrieval performance and answer quality. There are two ways to run evaluations:

#### Option A: Gradio Dashboard (Recommended)

Launch the interactive evaluation dashboard:

```bash
python src/evaluator.py
```

This opens a web interface with two main sections:

1. **Retrieval Evaluation**: Measures how well the system retrieves relevant documents
2. **Answer Evaluation**: Assesses the quality of generated answers using LLM-as-a-judge

The dashboard displays:
- **Color-coded metrics**: Green (excellent), Orange (good), Red (needs improvement)
- **Summary statistics**: Average scores across all test cases
- **Category breakdown**: Bar charts showing performance by question category
- **Real-time progress**: Progress bars during evaluation runs

**Thresholds:**
- **Retrieval metrics**: 
  - Green: MRR/nDCG ≥ 0.9, Coverage ≥ 90%
  - Orange: MRR/nDCG ≥ 0.75, Coverage ≥ 75%
- **Answer metrics** (1-5 scale):
  - Green: Score ≥ 4.5
  - Orange: Score ≥ 4.0

#### Option B: Command Line Interface

Run evaluation for a specific test case by index:

```bash
python src/evaluation/eval.py <test_row_number>
```

For example, to evaluate test case #0:
```bash
python src/evaluation/eval.py 0
```

This outputs detailed results including:
- Test question and reference answer
- Retrieval metrics (MRR, nDCG, keyword coverage)
- Generated answer from the RAG system
- LLM judge feedback and scores (accuracy, completeness, relevance)

#### Test Data Structure

Test cases are stored in `src/evaluation/tests.jsonl` (JSONL format). Each test includes:

- `question`: The question to evaluate
- `keywords`: List of keywords that should appear in retrieved documents
- `reference_answer`: Expected answer for comparison
- `category`: Question category (e.g., "direct_fact", "spanning", "temporal")

Example test case:
```json
{
  "question": "What is the main purpose of this document?",
  "keywords": ["purpose", "documentation", "usage"],
  "reference_answer": "This document provides information about...",
  "category": "direct_fact"
}
```

#### Evaluation Metrics Explained

**Retrieval Evaluation Metrics:**

1. **Mean Reciprocal Rank (MRR)**: Measures how quickly relevant documents appear in search results
   - Calculates 1/rank for each keyword's first occurrence
   - Averages across all keywords
   - Range: 0.0 to 1.0 (higher is better)
   - Example: If a keyword appears at rank 2, MRR contribution = 1/2 = 0.5

2. **Normalized Discounted Cumulative Gain (nDCG)**: Evaluates ranking quality considering position
   - Uses binary relevance (keyword found = 1, not found = 0)
   - Applies logarithmic discounting (documents at higher ranks contribute more)
   - Normalized against ideal ranking
   - Range: 0.0 to 1.0 (higher is better)

3. **Keyword Coverage**: Percentage of expected keywords found in retrieved documents
   - Calculated as: (keywords_found / total_keywords) × 100%
   - Range: 0% to 100% (higher is better)
   - Indicates whether all relevant information is being retrieved

**Answer Evaluation Metrics (LLM-as-a-Judge):**

The system uses GPT-4.1-nano as an automated judge to evaluate answer quality on three dimensions:

1. **Accuracy** (1-5 scale): Factual correctness compared to reference answer
   - 5: Perfectly accurate, matches reference
   - 3: Acceptable, mostly correct with minor issues
   - 1: Wrong or contains significant factual errors

2. **Completeness** (1-5 scale): How thoroughly the answer addresses all aspects
   - 5: All information from reference answer included
   - 3: Covers main points but missing some details
   - 1: Very incomplete, missing key information

3. **Relevance** (1-5 scale): How directly the answer addresses the question
   - 5: Directly answers question with no extra information
   - 3: Mostly relevant but includes some tangential content
   - 1: Off-topic or contains irrelevant information

The LLM judge compares the generated answer to:
- The original question
- The reference answer
- The retrieved context documents

It provides structured feedback explaining its scoring rationale.

#### Understanding Results

**Retrieval Results:**
- High MRR/nDCG (≥0.9): System reliably finds relevant documents early in rankings
- High Coverage (≥90%): Most expected keywords are being retrieved
- Low scores: May indicate need to adjust retrieval parameters (k, chunk size) or embedding model

**Answer Results:**
- High scores across all dimensions: RAG system producing quality answers
- Low accuracy: Generated content may contain hallucinations or errors
- Low completeness: Answers may be too brief or missing important details
- Low relevance: System may be including unnecessary context or going off-topic

**Category Analysis:**
- Compare performance across different question types
- Identify which categories need improvement
- Helps guide targeted system optimization

#### Evaluation Architecture

The evaluation system consists of three main components:

1. **`src/evaluation/test.py`**: Loads and parses test cases from JSONL files
2. **`src/evaluation/eval.py`**: Core evaluation logic with two main functions:
   - `evaluate_retrieval()`: Tests document retrieval performance
   - `evaluate_answer()`: Uses LLM-as-a-judge to evaluate answer quality
3. **`src/evaluator.py`**: Gradio dashboard that orchestrates evaluations and visualizes results

The evaluation functions integrate with the main RAG system (`src/retrieve.py`) to ensure consistency with actual system behavior.

## Configuration

Key configuration parameters in `src/retrieve.py`:
- `MODEL`: LLM model name (default: "gpt-4.1-nano")
- `db_name`: Vector database directory (default: "vector_db")
- `RETRIEVAL_K`: Number of documents to retrieve (default: 5)

Key configuration parameters in `src/ingest.py`:
- `chunk_size`: Document chunk size (default: 1500)
- `chunk_overlap`: Overlap between chunks (default: 200)
- Embedding model: `text-embedding-3-large`

## How It Works

1. **Document Processing**: Documents are loaded, split into chunks, and embedded using OpenAI embeddings
2. **Vector Storage**: Chunks are stored in ChromaDB with their embeddings
3. **Query Processing**: When a question is asked:
   - The question (combined with conversation history) is embedded
   - Similar chunks are retrieved from the vector database
   - Retrieved context is formatted into a system prompt
   - The LLM generates an answer using the context and conversation history

## Dependencies

Key dependencies include:
- `langchain` & `langchain-openai`: LLM and RAG framework
- `langchain-chroma`: ChromaDB integration
- `chromadb`: Vector database
- `openai`: OpenAI API client
- `gradio`: Web interface framework
- `tiktoken`: Token counting utilities

See `pyproject.toml` for the complete list of dependencies.

## Notes

- The vector database is persisted to disk, so you only need to run ingestion once (or when documents change)
- The system uses conversation history to provide context-aware answers
- Evaluation framework helps assess and improve system performance


