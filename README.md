# AI Document Assistant — Local RAG Application

A document question-answering application built with Retrieval-Augmented Generation (RAG).

The application enables users to upload PDF documents, ask questions in natural language, and receive answers generated from relevant passages retrieved from those documents.

The application runs locally using Ollama, so it does not require a paid LLM API.

## Features

- Upload PDF documents
- Extract text and preserve page metadata
- Split document text into overlapping chunks
- Generate local embeddings using `nomic-embed-text`
- Retrieve relevant chunks using FAISS vector search
- Generate answers using `llama3.2:3b`
- Display source filenames and page references
- Filter retrieved chunks using a configurable similarity threshold
- Return a not-found response when information is unavailable
- Run automated retrieval and answer checks
- Save evaluation results to a JSON report

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Ollama
- Llama 3.2 3B
- Nomic Embed Text
- FAISS
- NumPy

## Architecture

```text
PDF Upload
    |
    v
PDF Text Extraction
    |
    v
Text Chunking + Page Metadata
    |
    v
Local Embeddings
    |
    v
FAISS Vector Index
    |
    v
User Question
    |
    v
Question Embedding + Similarity Search
    |
    v
Relevant Document Chunks
    |
    v
Local Llama Model
    |
    v
Answer + Source References
```

## Prerequisites

Install the following before running the application:

- Python 3.10 or later
- Ollama for Windows

## Local AI Models

Download the required models using PowerShell:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify that both models are installed:

```powershell
ollama list
```

## Project Setup

Open a terminal inside the `demo1_pdf_rag` folder.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

Upload a PDF, ask a question, and view the generated answer and source references.

## Run the Evaluation

Run the automated evaluation:

```powershell
python evaluate_rag.py
```

The evaluation script reads test cases from `evaluation_cases.json` and saves detailed results to `evaluation_report.json`.

Ensure that the sample PDFs referenced by the evaluation dataset exist at their expected paths.

## Current Evaluation Results

In the current sample evaluation:

- Answer checks: 11/11 passed
- Retrieval checks: 8/8 passed
- Errors: 0
- Skipped tests: 0

These results apply only to the current test dataset and evaluation rules. They do not establish production-level accuracy or guarantee that hallucinations will never occur.

## Limitations

- This project is currently a local prototype.
- Similarity thresholds require further testing and tuning.
- Prompt instructions alone cannot guarantee factual correctness.
- Evaluation currently uses expected-text matching.
- Source display behavior depends partly on the generated answer and is not a complete factual verification system.
- Production security, authentication, and deployment hardening have not yet been implemented.

## Future Improvements

- Improve retrieval and answer evaluation
- Add more test questions and documents
- Improve answer verification and citation accuracy
- Add FastAPI backend integration
- Add logging and monitoring
- Explore Docker and cloud deployment

## Author

**Jotsana**

Generative AI Engineering Portfolio Project
