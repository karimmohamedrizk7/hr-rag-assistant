import os
import shutil
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ingest import load_documents, split_documents

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
DB_DIR = "chroma_db"
COLLECTION = "hr_policies"


def get_embeddings():
    """Step 3: the model that turns text into vectors."""
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def build_index(batch_size=20):
    """Step 4: embed all chunks and store them in Chroma."""
    chunks = split_documents(load_documents())
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")

    # start fresh so re-running does not create duplicates
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )
    for i in range(0, len(chunks), batch_size):
        db.add_documents(chunks[i : i + batch_size])
        print(f"  stored {min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(1)  # be gentle with the free API limits
    return db


def load_index():
    """Open the already-built database (used by the chatbot later)."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )


if __name__ == "__main__":
    db = build_index()

    query = "How many days of annual leave do I get?"
    print(f"\nTest query: {query}\n")
    for doc, score in db.similarity_search_with_score(query, k=3):
        print(f"[{score:.3f}] {doc.metadata['source']}")
        print(doc.page_content[:200].replace("\n", " "), "...\n")