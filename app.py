# The chatbot. Owners: Engineer 2 (retrieve + answer), Engineer 3 (Gradio + deploy)
# Step 5: Retrieve the 5 closest passages
# Step 6: Send question + passages to Gemini, return answer with sources
# Step 7: Gradio chat window
#
# Try it from the terminal:
#   python app.py "Can I get accessible housing?"
# Shows the passages found. Also asks Gemini if GEMINI_API_KEY is set: on a laptop in
# .env (copy .env.example), in Colab Secrets, or in Hugging Face Space Secrets. Never in code.

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from .env if the file exists; does nothing otherwise

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "chroma_db"                                    # built by ingest.py
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # MUST match ingest.py
TOP_K = 5
GEMINI_MODEL = "gemini-3.5-flash-lite"                     # change here if the team picks anothe
SDC_CONTACT = "the Student Disability Center (sdc.ucdavis.edu)"

_db = None


def get_db():
    """Open the search library once and reuse it for every question."""
    global _db
    if _db is None:
        if not os.path.isdir(DB_DIR):
            raise SystemExit(f"No {DB_DIR}/ found. Run ingest.py first.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return _db


# Step 5: Retrieve the 5 closest passages
def retrieve(question, k=TOP_K):
    """Return the k passages closest in meaning to the question.
    Each item is (passage, distance); a smaller distance means a closer match."""
    return get_db().similarity_search_with_score(question, k=k)


def format_passages(passages):
    """Number each passage and label it with its title and link, for the prompt."""
    blocks = []
    for i, (doc, _distance) in enumerate(passages, start=1):
        title = doc.metadata.get("title", "Untitled")
        source = doc.metadata.get("source", "unknown source")
        blocks.append(f"[{i}] {title} ({source})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def build_prompt(question, passages):
    """The instructions Gemini follows. The rules are what keep answers trustworthy."""
    return f"""You answer UC Davis students' questions about disability accommodations.

Rules:
- Use ONLY the passages below. Do not use outside knowledge.
- After each fact, cite the passage number it came from, like [1] or [2].
- If the passages do not contain the answer, say "I don't know" and suggest contacting {SDC_CONTACT}.
- Do not give personal advice or decide anything about a specific student's case.
- Keep the answer short and plain.

Passages:
{format_passages(passages)}

Question: {question}
Answer:"""


def list_sources(passages):
    """Unique title + link for each passage used, in order, for showing under the answer."""
    seen, sources = set(), []
    for doc, _distance in passages:
        key = (doc.metadata.get("title", "Untitled"), doc.metadata.get("source", ""))
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources


# Step 6: Send question + passages to Gemini, return answer with sources
def answer(question):
    """Return {"answer": text, "sources": [(title, link), ...]}. Step 7 calls this."""
    passages = retrieve(question)
    if not passages:
        return {"answer": f"I don't know. Please contact {SDC_CONTACT}.", "sources": []}

    from google import genai  # imported here so retrieval works without the Gemini library set up
    client = genai.Client()   # reads GEMINI_API_KEY from the environment
    response = client.models.generate_content(model=GEMINI_MODEL,
                                              contents=build_prompt(question, passages))
    return {"answer": response.text.strip(), "sources": list_sources(passages)}


# Step 7: Gradio chat window (Engineer 3) goes here; it should call answer(question).



# allows me to test from terminal
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I get accessible housing?"
    print(f"Question: {question}\n")
    print("Passages found (smaller distance = closer match):")
    for doc, distance in retrieve(question):
        snippet = " ".join(doc.page_content.split())[:120]
        print(f"  {distance:.3f}  {doc.metadata.get('title', 'Untitled')}: {snippet}...")

    key = os.environ.get("GEMINI_API_KEY", "")
    if key and key != "paste-your-key-here":
        result = answer(question)
        print("\nAnswer:\n" + result["answer"])
        print("\nSources:")
        for title, link in result["sources"]:
            print(f"  - {title}: {link}")
    else:
        print("\n(Put your key in .env to also get Gemini's answer.)")
