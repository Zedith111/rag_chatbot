# Multi-Modal RAG Chatbot

## Overview
A chatbot that integrates Retrieval-Augmented Generation(RAG) to provide contextual and knowledge-based response. Implement multi modal support, enabling better data retreival from texts, images and tables inside document. Build with ollama, langchain and streamlit

## Installation
### Prerequisites
- Python 3.12
- Ollama
- Docker (optional)
- poppler, tesseract, libmagic(For local installation without docker, use by unstructured)

### Set Up Instruction
For docker
```sh
docker compose up
```

For local
```sh
cd app
pip install -r requirements.txt
streamlit run app.py
```

After installation, streamlit app will be available at localhost:8501

## Reference
- [Multi-Modal RAG: A Practical Guide](https://gautam75.medium.com/multi-modal-rag-a-practical-guide-99b0178c4fbb)
- [Multimodal RAG: Chat with PDFs (Images & Tables)](https://www.youtube.com/watch?v=uLrReyH5cu0)