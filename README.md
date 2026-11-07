# Instamart Customer Intelligence & Support Copilot

## Overview

This project implements a customer intelligence and support copilot using synthetic Instamart data. It combines data cleaning, customer segmentation, lapse prediction, support-ticket classification, policy retrieval, and a tool-using refund agent.

The implementation uses Python, pandas, NumPy, scikit-learn, SQLite, joblib, and the supplied offline `mock_llm.py`. No paid API or external hosted LLM is used.

The project has four parts:

- **Part A:** Data audit, cleaning, validation, SQLite database creation, and SQL analysis.
- **Part B:** RFM analysis, customer segmentation, lapse prediction, and retention prioritisation.
- **Part C:** PII redaction, zero-shot/few-shot ticket classification, retry handling, and routing.
- **Part D:** TF-IDF policy retrieval and a resumable refund agent with safety limits.

All customer and transaction records are synthetic and fictitious.

## Reproduce

Run the following commands from the project root using Windows PowerShell.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate synthetic data:

   ```powershell
   python generate_data.py
   ```

5. Clean and validate the data and create the analytical database:

   ```powershell
   python part_a.py
   ```

6. Build the operational database used by the refund-agent scenarios:

   ```powershell
   python build_database.py
   ```

7. Run customer segmentation and lapse prediction:

   ```powershell
   python part_b.py
   ```

8. Run support-ticket classification and routing:

   ```powershell
   python part_c.py
   ```

9. Run policy chunking, TF-IDF retrieval, and policy evaluation:

   ```powershell
   python part_d.py
   ```

10. Run the refund-agent scenarios and recovery demonstrations:

    ```powershell
    python refund_agent.py
    ```

The supplied `mock_llm.py` must remain unmodified. All model interactions use this offline mock implementation.

## Part A

### Data audit and cleaning

Synthetic raw datasets include customers, products, orders, support tickets, agent scenarios, and policy evaluations.

The raw order dataset contained 13,084 rows, including 119 exact duplicate rows. Cleaning retained the latest record for each distinct order ID, resulting in 12,309 clean orders.

### Data validation

The cleaned datasets passed all 25 validation checkpoints.

| Dataset | Processed rows |
|---|---:|
| Customers | 1,200 |
| Orders | 12,965 |
| Products | 40 |
| Tickets | 164 |
| Agent scenarios | 7 |
| Policy evaluations | 14 |

The final clean orders dataset contains 12,309 distinct order IDs.

Additional validation results:

- Customer duplicate IDs: 0
- Product duplicate IDs: 0
- Ticket duplicate IDs: 0
- Exact duplicate order rows: 0
- Orders with unknown customers: 0
- Orders with unknown products: 0
- Tickets with unknown customers: 0
- Negative delivery times after cleaning: 0

Missing values were retained where appropriate or handled during cleaning.

### SQLite databases

Two databases are maintained for separate purposes:

- `data/instamart.db`: analytical database containing customers, products, and clean orders.
- `database/instamart.db`: operational database containing customers, products, order events, tickets, agent scenarios, and policy evaluations.

The operational database passed SQLite integrity and foreign-key validation.

### SQL and pandas analysis

SQL queries and pandas analysis were used to examine order status, revenue, customer activity, product/category performance, and other business metrics.

Clean order status distribution:

| Status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Total | 12,309 |

Completed-order revenue: **INR 2,998,668.80**.

## Part B

### RFM analysis

The RFM snapshot date is 2025-10-01.

| Metric | Result |
|---|---:|
| Customers in RFM analysis | 1,175 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 |
| Maximum frequency | 33 |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

Output: `outputs/rfm_table.csv`.

### Customer segmentation

K-Means clustering was evaluated for K values from 2 through 8 using StandardScaler, `n_init=10`, and `random_state=42`.

K=4 was selected based on silhouette score and business interpretability. Lower inertia alone was not used as the selection criterion because inertia decreases as the number of clusters increases.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Four customer segments:

| Segment ID | Customers | Mean recency | Mean frequency | Mean monetary |
|---:|---:|---:|---:|---:|
| 0 | 436 | 28.86 | 3.77 | INR 1,057.62 |
| 1 | 138 | 24.12 | 6.77 | INR 7,129.70 |
| 2 | 419 | 8.77 | 19.40 | INR 3,410.02 |
| 3 | 182 | 109.04 | 2.20 | INR 685.99 |

There are 138 Premium customers.

PCA explained variance:

- PC1: 0.634
- PC2: 0.218
- Combined: 0.852

Output: `outputs/customer_segments.csv`.

### Lapse prediction

The lapse cohort uses a cutoff date of 2025-09-01.

| Metric | Result |
|---|---:|
| Cohort size | 1,156 |
| Lapsed customers | 443 |
| Non-lapsed customers | 713 |
| Lapse rate | 38.32% |
| Training rows | 924 |
| Test rows | 232 |

Eleven features were engineered using historical customer/order data. Features were restricted to information available before the cutoff date.

The preprocessing pipeline was fitted on training data only. The test set was not used to fit the encoder, scaler, or resampling procedure.

### Model comparison

| Model | Test accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | — | 0.6444 | 0.6517 | 0.6480 |
| Random Forest | — | 0.6404 | 0.6404 | 0.6404 |

The baseline accuracy was 0.6164, with positive-class F1 of 0.

Logistic Regression was selected based on its better untouched test F1 score, despite Random Forest having a higher cross-validation F1 score.

### Training-only oversampling

| Metric | Original training data | Balanced training data |
|---|---:|---:|
| Precision | 0.6444 | 0.6000 |
| Recall | 0.6517 | 0.8090 |
| F1 | 0.6480 | 0.6890 |

Oversampling was performed only on the training data.

### Feature importance and retention list

Random Forest's top five feature importances:

| Feature | Importance |
|---|---:|
| Monetary | 0.277830 |
| Frequency | 0.276787 |
| Recency | 0.111219 |
| Average delivery minutes | 0.061649 |
| Average discount percentage | 0.053178 |

The saved Logistic Regression model was reloaded and its predictions were verified.

Outputs:

- `outputs/kmeans_evaluation.csv`
- `outputs/lapse_cohort.csv`
- `outputs/lapse_features.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/top5_feature_importances.csv`

The retention list is illustrative. Premium labels were calculated using the full-window RFM snapshot, which includes September. Therefore, the Premium flag in the retention list is not a leakage-free historical feature.

## Part C

### PII redaction

Tickets were processed in input order. Redaction was performed before any model call.

| PII type | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total | 278 |

A post-redaction check found zero remaining matches for the specified sensitive patterns.

The model received only the redacted ticket text as its user message.

### Guardrails and output validation

| Status | Tickets |
|---|---:|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |
| Total | 164 |

Empty tickets and tickets estimated above 400 tokens were excluded from model calls.

Every model call used a maximum of three retries with exponential backoff and jitter. The deterministic mock generated 19 transient errors, all recovered successfully.

### Zero-shot and few-shot comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 11,116 | 18,916 |
| Tokens per classified ticket | 75.62 | 128.68 |

Few-shot classification improved accuracy by 14.29 percentage points, at the cost of approximately 70.2% more tokens.

The zero-shot confusion matrix, with category order billing, delivery, fraud, technical:

```text
[[38,  0,  0,  0],
 [ 0, 36,  0,  0],
 [11,  0, 23,  0],
 [10,  0,  0, 29]]
```

### Token budget demonstration

The zero-shot budget demonstration used a configured token budget of 6,000.

- Tokens consumed: 6,010
- Budget-exhausted tickets: 72

The implementation checks the running total before each model call. Consequently, the final response can take the total slightly above the configured budget. This is a pre-call budget guard, not a strict post-response hard cap.

### Final routing

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |
| Total | 164 |

Invalid, empty, long, and budget-exhausted tickets are sent for human review. Fraud tickets are routed to the security team. Premium customers with otherwise valid classifications are routed to the premium desk.

Outputs:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

The final triage output contains exactly 164 rows and eight required columns.

## Part D

### Policy retrieval

Five policy documents are chunked into 40-word sections with a 10-word overlap.

| Policy | Chunks |
|---|---:|
| Refund | 4 |
| Delivery | 3 |
| Escalation | 3 |
| Payment | 3 |
| Privacy | 2 |
| Total | 15 |

A `TfidfVectorizer` is fitted on the policy chunks. Query vectors are transformed using the same fitted vectorizer, and cosine similarity ranks candidate chunks.

A lexical relevance check rejects unrelated retrieval results. Unsupported questions receive a fallback response instead of an invented answer.

### Policy evaluation

| Evaluation | Result |
|---|---:|
| Answerable questions | 12 |
| Unsupported questions | 2 |
| Total passed | 14/14 |

The zero-overlap test generated 13 chunks and successfully retrieved the refund policy. Both unsupported questions correctly triggered the fallback.

### Refund agent

The refund agent uses:

- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

Order data is retrieved from `data/instamart.db`. The seven agent scenarios are loaded from `database/instamart.db`.

### Seven-scenario evaluation

| Scenario | Expected result | Outcome |
|---|---|---|
| S1 | REFUND — INR 128.25 | PASS |
| S2 | REFUND — INR 135.00 | PASS |
| S3 | ESCALATE — INR 1,999.00 | PASS |
| S4 | DENIED | PASS |
| S5 | DENIED | PASS |
| S6 | NOT_FOUND | PASS |
| S7 | FALLBACK_MAX_ITERATIONS | PASS |

Refunds above INR 1,000 are escalated for human approval rather than automatically issued.

### Stop-safely and recovery tests

| Test | Expected result | Actual result |
|---|---|---|
| Token limit of 200 | FALLBACK_TOKEN_LIMIT | PASS |
| Time limit of 0 seconds | FALLBACK_TIMEOUT | PASS |
| Repeated tool loop | FALLBACK_MAX_ITERATIONS | PASS |
| Crash after second tool | Checkpoint saved | PASS |
| Resume | Continue from saved state | PASS |

The crash/resume demonstration executed exactly these tools across both runs:

```text
get_order
search_policy
calculate_refund
issue_refund
```

No completed tool was repeated.

### Refund ledger and execution logs

The final refund ledger contains two records:

| Order ID | Refund |
|---|---:|
| ORD-100048 | INR 128.25 |
| ORD-100067 | INR 135.00 |

Duplicate refund order IDs: 0.

The agent execution log contains 10 records: seven scenario executions and three stop-safely demonstrations. Checkpoints are stored in `state/`.

Outputs:

- `outputs/refund_ledger.jsonl`
- `outputs/agent_runs.jsonl`
- `state/`

### ReAct versus Plan-and-Execute

The implemented refund agent uses a ReAct-style loop: the mock LLM observes tool results and chooses the next action. This supports dynamic decisions when an order is missing, cancelled, eligible, or requires escalation.

A Plan-and-Execute agent would first create a fixed action plan and then execute its steps. It can reduce repeated planning calls for predictable workflows, but unexpected tool observations may require replanning.

The implemented ReAct S1 execution used 960 mock tokens across four tool executions. No separate Plan-and-Execute implementation was benchmarked, so a numerical token comparison is not claimed. For this prototype, ReAct is retained because the workflow must respond to actual database and policy observations.

## Limitations and Next Steps

- The datasets are synthetic and do not represent real customer behaviour.
- The mock LLM is deterministic and offline; its accuracy does not establish real-world LLM performance.
- TF-IDF retrieval is lexical and may miss semantically related policy passages. The lexical relevance filter is a lightweight safeguard, not a general semantic entailment system.
- The refund ledger simulates issuing refunds; it does not connect to a real payment gateway.
- Human escalation is simulated and does not create a real support ticket.
- The token budget check occurs before a model call, so the final response can exceed the configured limit slightly.
- The Premium retention list uses a full-window segment label and should not be interpreted as a leakage-free prospective ranking.
- Production deployment would require authentication, role-based permissions, secure audit logging, real payment-provider integration, stronger idempotency guarantees, monitoring, and human approval workflows.