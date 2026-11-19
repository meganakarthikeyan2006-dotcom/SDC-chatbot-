# Builds the search library. Owner: Engineer 1
# Step 1: Load the SDC documents listed in sources.csv (web pages + files in data/)
# Step 2: Chunk them into passages
# Step 3: Embed each passage with the Hugging Face model
# Step 4: Store everything in Chroma (chroma_db/)

import csv
import re
import shutil
from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
SOURCES_CSV = Path("sources.csv")
PERSIST_DIR = Path("chroma_db")
CACHE_DIR = DATA_DIR / ".cache"   # last known-good text for each fetched URL

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def looks_blocked(text):
    # Cloudflare/bot-challenge pages ("Just a moment...") return tiny, JS-only HTML
    return len(text.strip()) < 200 or "just a moment" in text.lower()


def cache_path_for(url):
    return CACHE_DIR / (re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:120] + ".txt")


# Loads Individual Source
def load_source(title, link):
    if link.startswith("http://") or link.startswith("https://"):
        docs = WebBaseLoader(link).load()
        cache_path = cache_path_for(link)
        if not docs or looks_blocked(docs[0].page_content):
            if cache_path.exists():
                print(f"Warning: '{title}' looks blocked (bot challenge?) - using last known-good cached copy")
                docs = [Document(page_content=cache_path.read_text(encoding="utf-8"), metadata={"source": link})]
            else:
                print(f"Skipping '{title}': fetch looks blocked (bot challenge?) and no cached copy to fall back on")
                return []
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(docs[0].page_content, encoding="utf-8")
    else:
        path = DATA_DIR / link
        loader_cls = LOADERS_BY_SUFFIX.get(path.suffix.lower())
        if not path.exists():
            print(f"Skipping '{title}': file not found at {path}")
            return []
        if loader_cls is None:
            print(f"Skipping '{title}': unsupported file type {path.suffix}")
            return []
        docs = loader_cls(str(path)).load()

    for doc in docs:
        doc.metadata["title"] = title
    return docs

# Loads all sources from sources.csv
def load_all_sources():
    docs = []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title, link = row["title"].strip(), row["link"].strip()
            if not link:
                continue
            docs.extend(load_source(title, link)) # uses previously defined function
    return docs


docs = load_all_sources()                                # 1. load + strip HTML/PDF text

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)                   # 2. chunk (metadata kept)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if PERSIST_DIR.exists():                                  # rebuild from scratch each run,
    shutil.rmtree(PERSIST_DIR)                             # so old/removed sources don't linger

db = Chroma.from_documents(chunks, embeddings,             # 3. embed + store
                           persist_directory=str(PERSIST_DIR))

print(f"Ingested {len(docs)} document(s) -> {len(chunks)} chunk(s) -> {PERSIST_DIR}/")
