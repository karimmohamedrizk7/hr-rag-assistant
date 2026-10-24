import csv
import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from build_index import load_index
from rag import LLM_MODEL, PROMPT, format_context, retrieve

load_dotenv()

PAUSE = 13  # seconds between LLM calls (free tier allows only a few per minute)

# question, file that should be retrieved, words that must appear in the answer
TEST_SET = [
    ("How many days of annual leave do I get?", "04_annual_leave.txt", ["21"]),
    ("How much is the overtime rate on Fridays?", "11_overtime.txt", ["200"]),
    ("How many days per week can I work remotely?", "08_remote_work.txt", ["2"]),
    ("What percentage of salary is paid during the first 90 days of sick leave?", "05_sick_leave.txt", ["75"]),
    ("How many days of maternity leave are given?", "06_parental_leave.txt", ["120"]),
    ("What is the notice period for managers who resign?", "22_resignation_and_termination.txt", ["60"]),
    ("On which day of the month are salaries paid?", "09_salary_payroll.txt", ["25"]),
    ("What is the yearly limit of the health insurance per person?", "12_health_insurance.txt", ["250,000"]),
    ("What is the annual training budget per employee?", "18_training_and_development.txt", ["10,000"]),
    # out-of-scope question: the bot must refuse instead of guessing
    ("Does the company allow pets in the office?", None, ["could not find"]),
]

JUDGE_PROMPT = """You are a strict grader.
Context:
{context}

Answer:
{answer}

Is every factual claim in the Answer supported by the Context?
Reply with only YES or NO."""


def ask(llm, messages, tries=6):
    """Call the LLM, wait after each call, and back off if rate-limited."""
    for attempt in range(tries):
        try:
            text = llm.invoke(messages).text
            time.sleep(PAUSE)
            return text
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"  limit/busy ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Gemini is unavailable right now, try again later.")


def main():
    db = load_index()
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, timeout=120, max_retries=1)
    rows = []

    for i, (question, expected_file, keywords) in enumerate(TEST_SET, 1):
        docs = retrieve(db, question)
        context = format_context(docs)
        retrieved = [os.path.basename(d.metadata["source"]) for d in docs]

        answer = ask(llm, PROMPT.format_messages(context=context, question=question))
        verdict = ask(llm, JUDGE_PROMPT.format(context=context, answer=answer))

        relevance = True if expected_file is None else expected_file in retrieved
        correct = all(k.lower() in answer.lower() for k in keywords)
        faithful = verdict.strip().upper().startswith("YES")

        rows.append([i, question, relevance, faithful, correct, answer.replace("\n", " ")])
        print(f"Q{i}: relevance={relevance} faithful={faithful} correct={correct}")

    n = len(rows)
    print("\n=== Summary ===")
    print(f"Retrieval relevance : {sum(r[2] for r in rows)}/{n}")
    print(f"Faithfulness        : {sum(r[3] for r in rows)}/{n}")
    print(f"Answer correctness  : {sum(r[4] for r in rows)}/{n}")

    with open("eval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["#", "question", "relevance", "faithful", "correct", "answer"])
        w.writerows(rows)
    print("Saved eval_results.csv")


if __name__ == "__main__":
    main()