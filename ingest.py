from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(folder="data"):
    """Step 1: Load all .txt files from the data folder."""
    loader = DirectoryLoader(
        folder,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_documents(docs, chunk_size=500, chunk_overlap=100):
    """Step 2: Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("\n--- Example chunk ---")
    print(chunks[5].page_content)
    print("\nMetadata:", chunks[5].metadata)