import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from build_index import load_index

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
TOP_K = 4

# Step 6 (Augmentation): the prompt that forces the model to stay grounded
PROMPT = ChatPromptTemplate.from_template(
    """You are the HR assistant of Nile Industries Co.
Answer the employee's question using ONLY the context below.

Rules:
- If the answer is not in the context, say exactly: "I could not find this in the HR policies. Please contact HR at extension 300."
- Do not use outside knowledge and do not guess.
- Be short and clear, and include the exact numbers from the context.
- Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""
)


def retrieve(db, question, k=TOP_K):
    """Step 5 (Retrieval): find the k most similar chunks."""
    return db.similarity_search(question, k=k)


def format_context(docs):
    return "\n\n---\n\n".join(
        f"[{os.path.basename(d.metadata['source'])}]\n{d.page_content}" for d in docs
    )


def answer(db, llm, question):
    """Step 7 (Generation): send question + context to the LLM."""
    docs = retrieve(db, question)
    messages = PROMPT.format_messages(
        context=format_context(docs), question=question
    )
    response = llm.invoke(messages)
    sources = sorted({os.path.basename(d.metadata["source"]) for d in docs})
    return response.text, sources


if __name__ == "__main__":
    db = load_index()
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, temperature=0, timeout=120, max_retries=3
    )

    print("HR Assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        text, sources = answer(db, llm, question)
        print(f"\nAssistant: {text}")
        print(f"Sources: {', '.join(sources)}\n")