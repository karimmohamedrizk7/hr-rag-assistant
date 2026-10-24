
# HR Policy Assistant (RAG)

A Retrieval-Augmented Generation chatbot that answers employee questions
about the HR policies of a fictional company (Nile Industries Co.), using
only the retrieved policy documents.

## Tech Stack
- Python 3.12
- LangChain (orchestration)
- Chroma (vector database)
- Google Gemini API (embeddings + LLM)

## Project Structure
- `data/` : 22 HR policy documents (.txt)
- `ingest.py` : loads documents and splits them into chunks
- `build_index.py` : creates embeddings and stores them in Chroma
- `rag.py` : retrieval, prompt, generation, and interactive chat (CLI)
- `evaluate.py` : automatic evaluation on test questions

## Setup
1. Create and activate a virtual environment:
```
   py -3.12 -m venv venv
   venv\Scripts\activate
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Get a free API key from Google AI Studio and create a `.env` file:
```
   GOOGLE_API_KEY=your_key_here
   LLM_MODEL=gemini-3.5-flash-lite
   EMBED_MODEL=gemini-embedding-001
```
   The `.env` file is listed in `.gitignore` and must never be committed.

## How to Run
1. Build the vector index (once):
```
   python build_index.py
```
2. Start the chatbot:
```
   python rag.py
```
3. (Optional) Run the evaluation:
```
   python evaluate.py
```
   Results are saved to `eval_results.csv`.

## How It Works
1. Documents are loaded and split into chunks (500 characters, 100 overlap).
2. Each chunk is converted to a vector with Gemini embeddings and stored in Chroma.
3. For each question, the top 4 most similar chunks are retrieved.
4. The chunks and the question are inserted into a strict prompt that tells
   the model to answer only from the context.
5. If the answer is not in the documents, the bot says it could not find it.

## Notes
- The free API tier has rate limits. `evaluate.py` pauses between calls and retries when limited.
- Model names can change. Edit `LLM_MODEL` and `EMBED_MODEL` in `.env` if a model is not found.