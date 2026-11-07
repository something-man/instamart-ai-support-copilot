
from pathlib import Path
import sqlite3
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "data" / "policies"
DB_PATH = BASE_DIR / "database" / "instamart.db"

CHUNK_SIZE = 40
OVERLAP = 10
TOP_K = 3


# --------------------------------------------------
# POLICY CHUNKING
# --------------------------------------------------

def chunk_policy_file(file_path, chunk_size=40, overlap=10):
    text = file_path.read_text(encoding="utf-8")
    words = text.split()

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "source": file_path.name,
            "text": chunk_text,
            "start_word": start,
            "end_word": end
        })

        # Stop after the first chunk that reaches the end.
        if end >= len(words):
            break

    return chunks


def load_policy_chunks(chunk_size=40, overlap=10):
    all_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        all_chunks.extend(
            chunk_policy_file(file_path, chunk_size, overlap)
        )

    return all_chunks


# --------------------------------------------------
# TF-IDF RETRIEVAL
# --------------------------------------------------

class PolicyRetriever:

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No policy chunks found.")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()

        self.chunk_matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    
    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.chunk_matrix
        ).flatten()

        # Remove common question words.
        stop_words = {
            "what", "who", "when", "where", "why", "how",
            "is", "are", "do", "does", "can", "the", "a",
            "an", "of", "to", "for", "in", "on", "my",
            "i", "me", "and", "or", "must", "have"
        }

        query_terms = {
            word for word in re.findall(r"[a-z]+", query.lower())
            if word not in stop_words and len(word) > 2
        }

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            chunk_terms = set(
                re.findall(r"[a-z]+", chunk["text"].lower())
            )

            overlap = query_terms.intersection(chunk_terms)

            # Require meaningful lexical overlap.
            if len(overlap) < 2:
                continue

            if scores[index] <= 0:
                continue

            results.append({
                "source": chunk["source"],
                "score": float(scores[index]),
                "text": chunk["text"]
            })

            if len(results) >= k:
                break

        return results

# --------------------------------------------------
# POLICY ASSISTANT
# --------------------------------------------------

def answer_policy_question(question, retriever, k=3):

    results = retriever.retrieve(question, k=k)

    # No matching policy context: do not invent an answer.
    if not results:
        return {
            "answer": "I could not find that information in the policy documents.",
            "sources": [],
            "retrieved": []
        }

    import mock_llm

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer questions using only "
                "the supplied policy context. Do not invent facts."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    response = mock_llm.chat(messages)

    return {
        "answer": response.choices[0].message.content,
        "sources": list(dict.fromkeys(
            item["source"] for item in results
        )),
        "retrieved": results
    }


# --------------------------------------------------
# POLICY EVALUATION
# --------------------------------------------------

def load_policy_evaluations():
    conn = sqlite3.connect(DB_PATH)

    evaluations = pd.read_sql_query(
        "SELECT * FROM policy_eval ORDER BY evaluation_id",
        conn
    )

    conn.close()
    return evaluations


def evaluate_policy_assistant(retriever):

    evaluations = load_policy_evaluations()

    passed = 0
    results = []

    print("\n" + "=" * 65)
    print("POLICY ASSISTANT EVALUATION")
    print("=" * 65)

    for _, row in evaluations.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]
        missing = int(row["expected_fact_missing"])

        result = answer_policy_question(question, retriever)

        answer = result["answer"]
        sources = result["sources"]

        if missing:
            test_pass = (
                len(sources) == 0
                and "could not find" in answer.lower()
            )
        else:
            test_pass = (
                expected_doc in sources
                and expected_fact.lower() in answer.lower()
            )

        if test_pass:
            passed += 1

        results.append({
            "evaluation_id": int(row["evaluation_id"]),
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": answer,
            "retrieved_sources": ", ".join(sources),
            "passed": test_pass
        })

        status = "PASS" if test_pass else "FAIL"

        print(f"\nQ{row['evaluation_id']}: {question}")
        print("Answer:", answer)
        print("Sources:", sources)
        print("Result:", status)

    print("\nEvaluation score:", passed, "/", len(evaluations))

    return results


# --------------------------------------------------
# RETRIEVAL VALIDATION
# --------------------------------------------------


def test_zero_overlap():

    chunks = load_policy_chunks(
        chunk_size=40,
        overlap=0
    )

    retriever = PolicyRetriever(chunks)

    results = retriever.retrieve(
        "What refund is given for completed orders delivered more than 45 minutes?",
        k=3
    )

    print("\nZERO-OVERLAP TEST")
    print("Chunk count:", len(chunks))
    print("Retrieved chunks:", len(results))

    assert len(chunks) == 13, (
        f"Expected 13 zero-overlap chunks, got {len(chunks)}"
    )

    assert len(results) > 0, (
        "Zero-overlap retrieval failed."
    )

    assert any(
        item["source"] == "refund_policy.txt"
        for item in results
    ), "Expected refund policy was not retrieved."

    print("Zero-overlap retrieval: PASS")

def test_missing_information(retriever):

    questions = [
        "What is the favourite colour of the company mascot?",
        "Who won the cricket world cup final?"
    ]

    print("\nMISSING INFORMATION FALLBACK TEST")

    for question in questions:

        result = answer_policy_question(question, retriever)

        assert not result["sources"], (
            f"Unexpected policy source for: {question}"
        )

        assert "could not find" in result["answer"].lower()

        print("Question:", question)
        print("Fallback:", result["answer"])
        print("PASS")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("LOADING POLICY DOCUMENTS")

    chunks = load_policy_chunks(
        CHUNK_SIZE,
        OVERLAP
    )

    print("Policy files:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Total chunks:", len(chunks))

    for source in sorted(set(c["source"] for c in chunks)):
        count = sum(1 for c in chunks if c["source"] == source)
        print(f"{source}: {count} chunks")

    retriever = PolicyRetriever(chunks)

    print("\nTF-IDF index created successfully.")

    evaluation_results = evaluate_policy_assistant(retriever)

    test_zero_overlap()
    test_missing_information(retriever)

    print("\n" + "=" * 65)
    print("RAG POLICY ASSISTANT TESTS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
