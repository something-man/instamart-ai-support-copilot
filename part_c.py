from pathlib import Path
import json
import re
import random
import time

import pandas as pd
import mock_llm
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "tickets.csv"
SEGMENT_FILE = OUTPUT_DIR / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100000

BASE_DELAY = 0.05
MAX_RETRIES = 3

CATEGORIES = {"billing", "delivery", "technical", "fraud"}


# ============================================================
# PROMPTS
# ============================================================

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
billing, delivery, technical, fraud

Return JSON only in this exact form:
{"category": "...", "reason": "..."}
Do not include markdown or additional text.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
billing, delivery, technical, fraud

Return JSON only in this exact form:
{"category": "...", "reason": "..."}
Do not include markdown or additional text.

EXAMPLES:

Example 1:
Ticket: My payment was deducted twice.
Output: {"category":"billing","reason":"Duplicate payment"}

Example 2:
Ticket: Someone used my account without permission.
Output: {"category":"fraud","reason":"Unauthorized account activity"}

Example 3:
Ticket: My application crashes when I open it.
Output: {"category":"technical","reason":"Application crash"}
"""


# ============================================================
# LOAD DATA
# ============================================================

tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)

assert len(tickets) == 164
assert list(tickets.columns) == [
    "ticket_id", "customer_id", "order_id", "text", "true_category"
]


# ============================================================
# PII REDACTION
# ============================================================

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

redacted_texts = []
redaction_counts = {
    "emails": 0,
    "phones": 0,
    "order_ids": 0
}

for text in tickets["text"].astype(str):
    text, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phone_count = PHONE_PATTERN.subn("[PHONE]", text)
    text, order_count = ORDER_PATTERN.subn("[ORDER_ID]", text)

    redaction_counts["emails"] += email_count
    redaction_counts["phones"] += phone_count
    redaction_counts["order_ids"] += order_count

    assert EMAIL_PATTERN.search(text) is None
    assert PHONE_PATTERN.search(text) is None
    assert ORDER_PATTERN.search(text) is None

    redacted_texts.append(text)

tickets["redacted_text"] = redacted_texts

total_redactions = sum(redaction_counts.values())

print("\nPII REDACTION")
print("Emails:", redaction_counts["emails"])
print("Phones:", redaction_counts["phones"])
print("Order IDs:", redaction_counts["order_ids"])
print("Total:", total_redactions)
print("Remaining sensitive patterns: 0")

assert redaction_counts == {
    "emails": 55,
    "phones": 91,
    "order_ids": 132
}


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def est_tokens(text):
    return max(1, (len(text) + 3) // 4)


# ============================================================
# RETRY WRAPPER
# ============================================================

def call_with_retry(messages, ticket_id):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"[RETRY] Ticket {ticket_id} | "
                f"Retry #{retry_number} | Wait: {wait:.4f}s"
            )

            time.sleep(wait)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_response(response_text):

    try:
        result = json.loads(response_text)

        if (
            isinstance(result, dict)
            and result.get("category") in CATEGORIES
        ):
            return result["category"], "classified_json"

    except (json.JSONDecodeError, TypeError):
        pass

    words = re.findall(r"[a-z]+", str(response_text).lower())
    found = CATEGORIES.intersection(words)

    if len(found) == 1:
        return next(iter(found)), "classified_keyword"

    return "", "flagged_invalid_output"


# ============================================================
# FULL CLASSIFICATION PIPELINE
# ============================================================

def run_pipeline(prompt, token_budget=None):

    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        original_text = ticket["text"]
        redacted_text = ticket["redacted_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "category": "",
            "status": "",
            "tokens_used": 0
        }

        # Empty ticket guardrail
        if not original_text.strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        # Long ticket guardrail
        if est_tokens(original_text) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        # Budget guardrail
        if token_budget is not None and total_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        # Privacy assertion
        assert EMAIL_PATTERN.search(redacted_text) is None
        assert PHONE_PATTERN.search(redacted_text) is None
        assert ORDER_PATTERN.search(redacted_text) is None

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text}
        ]

        response = call_with_retry(messages, ticket_id)

        used_tokens = response.usage.total_tokens
        total_tokens += used_tokens

        category, status = validate_response(response.choices[0].message.content)

        result["category"] = category
        result["status"] = status
        result["tokens_used"] = used_tokens

        results.append(result)

    result_df = pd.DataFrame(results)

    return result_df, total_tokens


# ============================================================
# ZERO-SHOT AND FEW-SHOT COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("ZERO-SHOT CLASSIFICATION")
print("=" * 60)

zero_results, zero_tokens = run_pipeline(ZERO_SHOT_PROMPT)

zero_classified = zero_results[
    zero_results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
]

zero_accuracy = accuracy_score(
    zero_classified["true_category"],
    zero_classified["category"]
)

print("Classified:", len(zero_classified))
print("Accuracy:", round(zero_accuracy * 100, 2), "%")
print("Total tokens:", zero_tokens)
print(
    "Tokens per classified ticket:",
    round(zero_tokens / len(zero_classified), 2)
)

print("\nZERO-SHOT STATUS COUNTS")
print(zero_results["status"].value_counts().to_string())

print("\nZERO-SHOT CONFUSION MATRIX")
print(
    confusion_matrix(
        zero_classified["true_category"],
        zero_classified["category"],
        labels=sorted(CATEGORIES)
    )
)


print("\n" + "=" * 60)
print("FEW-SHOT CLASSIFICATION")
print("=" * 60)

few_results, few_tokens = run_pipeline(FEW_SHOT_PROMPT)

few_classified = few_results[
    few_results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
]

few_accuracy = accuracy_score(
    few_classified["true_category"],
    few_classified["category"]
)

print("Classified:", len(few_classified))
print("Accuracy:", round(few_accuracy * 100, 2), "%")
print("Total tokens:", few_tokens)
print(
    "Tokens per classified ticket:",
    round(few_tokens / len(few_classified), 2)
)

print("\nFEW-SHOT STATUS COUNTS")
print(few_results["status"].value_counts().to_string())


# ============================================================
# TOKEN BUDGET DEMONSTRATION
# ============================================================

print("\n" + "=" * 60)
print("TOKEN BUDGET DEMONSTRATION")
print("=" * 60)

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=TOKEN_BUDGET_DEMO
)

budget_path = OUTPUT_DIR / "triage_budget_demo.csv"

budget_results.to_csv(budget_path, index=False)

budget_exhausted = int(
    (budget_results["status"] == "budget_exhausted").sum()
)

print("Budget:", TOKEN_BUDGET_DEMO)
print("Tokens consumed:", budget_tokens)
print("Budget exhausted tickets:", budget_exhausted)
print("Saved:", budget_path)

if budget_exhausted == 0:
    print(
        "WARNING: No ticket exhausted the budget. "
        "Reduce TOKEN_BUDGET_DEMO and rerun."
    )


# ============================================================
# FINAL FEW-SHOT RUN AND CUSTOMER ROUTING
# ============================================================

print("\n" + "=" * 60)
print("FINAL TRIAGE AND ROUTING")
print("=" * 60)

final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=TOKEN_BUDGET_FINAL
)

segments = pd.read_csv(SEGMENT_FILE)

segment_lookup = segments[
    ["customer_id", "segment_name", "is_premium"]
].copy()

final_results = final_results.merge(
    segment_lookup,
    on="customer_id",
    how="left"
)

final_results["segment_name"] = (
    final_results["segment_name"].fillna("")
)

final_results["is_premium"] = (
    final_results["is_premium"].fillna(False).astype(bool)
)

def assign_route(row):

    if row["status"] in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        return pd.Series(["NORMAL", "human_review"])

    if row["category"] == "fraud":
        return pd.Series(["HIGH", "security_team"])

    if row["is_premium"]:
        return pd.Series(["HIGH", "premium_desk"])

    return pd.Series(["NORMAL", "support_queue"])


final_results[["priority", "route"]] = final_results.apply(
    assign_route,
    axis=1
)

final_output = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

assert len(final_output) == 164

final_path = OUTPUT_DIR / "triage_results.csv"

final_output.to_csv(final_path, index=False)

print("Final tickets:", len(final_output))
print("\nFinal status counts:")
print(final_output["status"].value_counts().to_string())

print("\nRouting counts:")
print(final_output["route"].value_counts().to_string())

print("\nSaved:", final_path)
print("\nPART C PIPELINE COMPLETED")