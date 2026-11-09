# SDC Accommodations Chatbot

A chatbot that answers UC Davis students' questions about Student Disability Center accommodations, using official SDC documents and showing its sources.

Built by AI Student Collective @ UC Davis, Fall 2026.

## How it works
1. `ingest.py` loads, chunks and embeds the SDC documents into a Chroma database.
2. `app.py` finds the most relevant passages for a question, sends them to Gemini, and shows the answer with sources in a Gradio chat window.

## Files
- `data/` – SDC documents
- `sources.csv` – where each document came from
- `ingest.py` – builds the search library
- `app.py` – the chatbot
- `requirements.txt` – libraries to install
- `eval/` – test questions

The API key is never committed. Use Colab Secrets while building and Hugging Face Space Secrets when deployed.
