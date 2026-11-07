# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains a four-part customer intelligence and support copilot built for the Instamart capstone. It uses synthetic, fictional customer, product, order, and support-ticket data to demonstrate a reproducible data pipeline, customer analytics, ticket triage, policy retrieval, and a tool-using refund agent.

The project is implemented with Python 3.11+, pandas, NumPy, scikit-learn, SQLite, and joblib. All language-model interactions are routed through the supplied offline `mock_llm.py`; no paid API, hosted LLM, or API key is required.

### Project components

- **Part A — Data Foundation:** raw-data audit, cleaning, validation, SQLite storage, SQL queries, and pandas analysis.
- **Part B — Customer Intelligence:** RFM analysis, K-Means segmentation, lapse prediction, model evaluation, and retention prioritisation.
- **Part C — Support Ticket Triage:** PII redaction, zero-shot/few-shot classification, retry handling, response validation, token-budget controls, and routing.
- **Part D — Policy RAG and Refund Agent:** policy chunking and TF-IDF retrieval, unsupported-question fallback, refund decision tools, checkpointing, and crash recovery.

All records and policies are synthetic examples created for this assignment. They are not real Instamart customer data or official company policies. Monetary values are in Indian Rupees (INR).

## Reproduce

Run these commands from the repository root in Windows PowerShell. Use Python 3.11 or newer and SQLite 3.25 or newer.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the pinned project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the synthetic raw datasets:

   ```powershell
   python generate_data.py
   ```

5. Run Part A data cleaning and create the analytical SQLite database:

   ```powershell
   python part_a.py
   ```

6. Run the formal Part A data and database validation:

   ```powershell
   python validate_data.py
   ```

7. Build the operational database used by the refund-agent scenarios:

   ```powershell
   python build_database.py
   ```

8. Validate the operational database:

   ```powershell
   python validate_database.py
   ```

9. Run the pandas exploratory analysis:

   ```powershell
   python pandas_analysis.py
   ```

10. Run Part B RFM analysis, segmentation, lapse prediction, and retention prioritisation:

    ```powershell
    python part_b.py
    ```

11. Run Part C ticket classification, guardrail, and routing experiments:

    ```powershell
    python part_c.py
    ```

12. Run Part D policy chunking, retrieval evaluation, and zero-overlap comparison:

    ```powershell
    python part_d.py
    ```

13. Run the refund-agent scenarios, stop-safely tests, and checkpoint recovery demonstration:

    ```powershell
    python refund_agent.py
    ```

`part_a.py` creates `data/clean/` and `data/instamart.db`. `build_database.py` creates the separate operational database at `database/instamart.db`. The supplied `mock_llm.py` is intended to be copied verbatim from the assignment and used without modification. The commands above reproduce the project outputs from the synthetic data generator.

## Part A

### Raw data audit

The raw datasets include customers, products, orders, tickets, agent scenarios, and policy-evaluation examples. The main raw table dimensions were:

| Dataset | Raw shape |
|---|---:|
| Customers | 1,200 × 5 |
| Products | 40 × 4 |
| Orders | 13,084 × 9 |
| Tickets | 164 × 5 |
| Agent scenarios | 7 × 5 |
| Policy evaluation | 14 × 3 |

The raw audit identified the following issues:

| Audit item | Raw result |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct `order_id` values | 12,309 |
| Non-numeric `order_amount` entries | 120 |
| Negative `delivery_minutes` values | 106 |
| Blank `delivery_minutes` values | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank customer genders | 30 |

Missing-value audit highlights:

| Dataset / column | Missing count | Percentage |
|---|---:|---:|
| Customers — age | 61 | 5.08% |
| Customers — gender | 30 | 2.50% |
| Orders — order_amount | 53 | 0.41% |
| Orders — delivery_minutes | 2,541 | 19.42% |
| Tickets — order_id | 8 | 4.88% |
| Tickets — text | 4 | 2.44% |
| Policy evaluation — expected_fact | 2 | 14.29% |

The raw audit also identified 119 exact duplicate order rows and repeated order IDs representing order-event/status history. The intermediate cleaned order-event dataset contained 12,965 rows after exact duplicate removal. The final analytical order table contains one latest record per distinct order ID: 12,309 rows. These are different stages of the pipeline and should not be confused.

### Cleaning rules

- Customer city values were trimmed and converted to title case, leaving exactly six standardised cities.
- Gender spellings were mapped to `male`, `female`, or `unknown`; blank values map to `unknown`.
- Ages outside 10–100 were treated as missing. Missing ages were imputed using the median age within each city, calculated with `groupby` and `transform`.
- Signup dates were converted to datetime values.
- Product names and categories were normalised, and product prices were converted to numeric values.
- Order timestamps and numeric fields were parsed; status values were normalised.
- Exact duplicate order rows were removed. Order events were sorted by order ID and timestamp, and the latest event was retained for each order ID.
- Missing order amounts were recalculated from product price, quantity, and discount where possible.
- Negative delivery times were treated as missing. Missing delivery times for completed orders were imputed using the product-category median; cancelled orders retain missing delivery times.
- Ticket text was read with `keep_default_na=False` so blank ticket messages remain empty strings during audit. The ticket text was not imputed as part of the Part A clean customer/product/order exports.

### Validation

The formal Part A validation completed with **25 of 25 checkpoints passed**.

| Validation | Result |
|---|---:|
| Clean customers | 1,200 |
| Clean products | 40 |
| Clean orders | 12,309 |
| Unique customer, product, and order IDs | PASS |
| Exactly six standardised cities | PASS |
| Customer ages complete and within 10–100 | PASS |
| No pending orders in final clean orders | PASS |
| No missing order amounts | PASS |
| Completed orders with missing delivery time | 0 |
| Cancelled orders with non-missing delivery time | 0 |
| Unknown customer/product foreign keys | 0 |
| SQLite integrity check | `ok` |
| SQLite foreign-key violations | 0 |
| Validation checkpoints | **25/25 PASS** |

Final clean order status distribution:

| Status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| **Total** | **12,309** |

Completed-order revenue is **INR 2,998,668.80**.

### SQLite databases

Two SQLite databases are used for separate purposes:

- `data/instamart.db` — analytical database with exactly three required tables: `customers`, `products`, and `orders`.
- `database/instamart.db` — operational database used by the refund-agent workflow; it contains customer, product, order-event, ticket, scenario, and policy-evaluation data, including the `current_orders` view.

The analytical database has 1,200 customers, 40 products, and 12,309 clean orders. The operational database passed its SQLite integrity and foreign-key checks.

### SQL analysis

The seven SQL queries in `sql/queries.sql` cover customer value, revenue, product pricing, inactive customers, above-average spend, city-level ranking, and cumulative daily revenue.

| Query | Finding |
|---|---|
| Q1 — High-value customers | 30 customers met the query conditions. Customer `C0693` had 6 qualifying orders and INR 13,776.55 in the query's spend measure. |
| Q2 — City/category revenue | Bengaluru/Electronics was the largest city-category combination at INR 233,909.45. |
| Q3 — Above-average product prices | 11 products were priced above the catalogue average. |
| Q4 — No completed orders | 25 customers had no completed orders. |
| Q5 — Above-average customer spend | 550 customers were above the average customer spend of INR 2,552.06. |
| Q6 — Top customer by city | The query ranked 1,175 customers; rank and row-number handling differed for 8 tied positions. |
| Q7 — Daily cumulative revenue | 183 daily rows covered 2025-04-01 to 2025-09-30; cumulative completed revenue was INR 2,998,668.80. |

The highest customer spend in the customer-level analysis was INR 14,766.55 for customer `C0693`.

### Pandas analysis

Pandas analysis independently aggregated revenue by city and product category and reconciled the result with SQL.

- Pandas pivot revenue: **INR 2,998,668.80**.
- SQL completed revenue: **INR 2,998,668.80**.
- Reconciliation: **match = True**.

City-level completed-order results:

| City | Revenue (INR) | Average order value (INR) | Orders |
|---|---:|---:|---:|
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Chennai | 463,629.15 | 272.72 | 1,700 |

Order-amount distribution:

| Statistic | Result |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| First quartile (Q1) | INR 64.80 |
| Third quartile (Q3) | INR 297.00 |
| Interquartile range (IQR) | INR 232.20 |
| Upper outlier fence | INR 645.30 |
| Values above upper fence | 897 |
| Values below lower fence | 0 |

The strongest absolute pairwise correlation reported was between order amount and discount percentage: **-0.095**, a very weak negative correlation.

## Part B

### RFM analysis

The RFM snapshot date is **2025-10-01**. RFM was calculated for customers with completed-order history.

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

### K-Means customer segmentation

K-Means was evaluated for K values 2–8 using `StandardScaler`, `n_init=10`, and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was selected because it had the highest tested silhouette score and produced interpretable customer profiles. Inertia was not used alone because it decreases as K increases.

| Segment ID | Customers | Mean recency | Mean frequency | Mean monetary (INR) |
|---:|---:|---:|---:|---:|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 |
| 3 | 182 | 109.04 | 2.20 | 685.99 |

Segment 1 was labelled Premium and contains **138 customers**.

PCA explained variance:

| Component | Explained variance |
|---|---:|
| PC1 | 0.634 |
| PC2 | 0.218 |
| Combined | 0.852 |

Output: `outputs/customer_segments.csv`.

### Lapse cohort and feature engineering

The lapse cohort uses a cutoff date of **2025-09-01**.

| Metric | Result |
|---|---:|
| Cohort customers | 1,156 |
| Lapsed customers | 443 |
| Non-lapsed customers | 713 |
| Lapse rate | 38.32% |
| Training rows | 924 |
| Test rows | 232 |
| Engineered features | 11 |
| Missing feature values | 0 |

The features include recency, frequency, monetary value, discount, cancellation, delivery, favourite category, city, age, gender, and tenure information. The preprocessing pipeline was fitted on training data only. The test set was not used to fit preprocessing or training-only resampling.

### Model training and evaluation

The baseline model achieved accuracy **0.6164** and positive-class F1 **0.0000**.

| Model | Test accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.7284** | 0.6444 | 0.6517 | **0.6480** |
| Random Forest | 0.7241 | 0.6404 | 0.6404 | 0.6404 |

Test confusion matrices:

```text
Logistic Regression:
[[111, 32],
 [ 31, 58]]

Random Forest:
[[111, 32],
 [ 32, 57]]
```

Best cross-validation F1:

| Model | Best parameters | CV F1 |
|---|---|---:|
| Logistic Regression | `C=100` | 0.6506 |
| Random Forest | `max_depth=8`, `min_samples_leaf=10` | 0.6865 |

Logistic Regression was selected based on its higher F1 on the held-out test set. Random Forest had the higher cross-validation F1; this difference is reported rather than hidden.

### Training-only class balancing

The original training class counts were `{0: 570, 1: 354}`. After training-only oversampling, the counts were `{0: 570, 1: 570}`.

| Metric | Original training data | Balanced training data |
|---|---:|---:|
| Precision | 0.6444 | 0.6000 |
| Recall | 0.6517 | 0.8090 |
| F1 | 0.6480 | **0.6890** |

Balanced-model confusion matrix:

```text
[[95, 48],
 [17, 72]]
```

Oversampling was applied only to the training data.

### Feature interpretation and retention list

Random Forest top-five feature importances:

| Feature | Importance |
|---|---:|
| Monetary | 0.277830 |
| Frequency | 0.276787 |
| Recency | 0.111219 |
| Average delivery minutes | 0.061649 |
| Average discount percentage | 0.053178 |

The three largest-magnitude Logistic Regression coefficients were frequency (-1.698239), favourite category Electronics (-1.136713), and favourite category Grocery (-0.448673). Negative coefficients indicate lower estimated lapse odds relative to the relevant baseline category in this fitted model; they do not establish causation.

The saved Logistic Regression model was reloaded and prediction reproduction passed. The retention list contains 10 Premium test customers; 3 were actually lapsed.

Outputs:

- `outputs/customer_segments.csv`
- `outputs/rfm_table.csv`
- `outputs/kmeans_evaluation.csv`
- `outputs/lapse_cohort.csv`
- `outputs/lapse_features.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/top5_feature_importances.csv`

The retention list is illustrative. Premium labels were calculated from the full-window RFM snapshot, which includes September. Therefore, the Premium flag in this list is not a leakage-free historical feature.

## Part C

### PII redaction

Ticket text was redacted before any model call. The redaction checks covered email addresses, phone numbers, and order IDs.

| PII type | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

The post-redaction assertion found zero remaining matches for the specified patterns. Empty and over-length tickets were excluded from model calls.

### Classification, retries, and output validation

The full ticket dataset contained 164 tickets. Final status counts:

| Status | Tickets |
|---|---:|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |
| **Total** | **164** |

The classifier uses a maximum of three retries with exponential backoff and jitter. In this deterministic mock run, retry number was 0 for each retry, producing waits in the 0.045–0.0575 second range. All **19 transient errors** were recovered.

Responses are parsed as JSON first; keyword fallback is used when JSON validation fails. Empty, long, invalid, and budget-exhausted cases are not treated as trusted model classifications and are routed to human review.

### Zero-shot and few-shot comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | **100.00%** |
| Total tokens | 11,116 | 18,916 |
| Tokens per classified ticket | 75.62 | 128.68 |

Few-shot classification improved accuracy by **14.29 percentage points**, with approximately **70.2% more tokens** per run.

Zero-shot confusion matrix (category order: billing, delivery, fraud, technical):

```text
[[38,  0,  0,  0],
 [ 0, 36,  0,  0],
 [11,  0, 23,  0],
 [10,  0,  0, 29]]
```

### Token-budget demonstration

The budget demonstration used a configured budget of 6,000 tokens.

| Metric | Result |
|---|---:|
| Configured budget | 6,000 |
| Tokens consumed | 6,010 |
| Budget-exhausted tickets | 72 |

The implementation checks the running token total before making each model call. The final response can therefore take the total slightly above the configured budget. This is a pre-call guard, not a strict post-response hard cap.

### Final routing

The final few-shot run used a token budget of 100,000. Routing results:

| Route | Tickets |
|---|---:|
| `support_queue` | 102 |
| `security_team` | 34 |
| `human_review` | 17 |
| `premium_desk` | 11 |
| **Total** | **164** |

Routing rules:

1. Empty, long, invalid-output, or budget-exhausted tickets go to `human_review` with normal priority.
2. Otherwise, fraud tickets go to `security_team` with high priority.
3. Otherwise, valid tickets belonging to Premium customers go to `premium_desk` with high priority.
4. Remaining valid tickets go to `support_queue` with normal priority.
5. Customers missing from the segment file are treated as non-Premium.

Outputs:

- `outputs/triage_results.csv` — 164 rows and 8 columns.
- `outputs/triage_budget_demo.csv` — records the budget-exhaustion demonstration.

## Part D

### Policy document chunking and retrieval

The policy assistant reads five text documents from `data/policies/`. Each document is split into 40-word chunks with a 10-word overlap. The first chunk begins at word 0; each next chunk begins 30 words after the previous start.

| Policy document | Chunks |
|---|---:|
| `refund_policy.txt` | 4 |
| `delivery_policy.txt` | 3 |
| `escalation_policy.txt` | 3 |
| `payment_policy.txt` | 3 |
| `privacy_policy.txt` | 2 |
| **Total** | **15** |

A `TfidfVectorizer` is fitted on the chunk text. Query text is transformed using the same fitted vectorizer, and cosine similarity ranks the candidate chunks. A lexical relevance check rejects unrelated retrieval results. If no relevant policy evidence is found, the assistant returns an explicit fallback rather than inventing an answer.

### RAG evaluation and overlap comparison

The current 10-word-overlap configuration passed all 14 evaluation questions.

| Evaluation | 10-word overlap | No overlap |
|---|---:|---:|
| Chunk count | 15 | 13 |
| Expected document hits | 12/12 | 12/12 |
| Expected fact hits | **12/12** | 9/12 |
| Unsupported-question fallbacks | 2/2 | 2/2 |
| Overall evaluation | **14/14** | 11/14 |

The no-overlap version retrieved the expected document for all 12 answerable questions, but failed to preserve the complete fact for three questions:

- The completed-order refund question lost the phrase **“more than 45 minutes.”**
- The refund-approval question lost the fact **“human support lead.”**
- The failed-payment question lost the **“48 hours”** return timeline.

This demonstrates that document-level retrieval success does not guarantee fact-level answer completeness. Overlap preserves context that would otherwise be split across chunk boundaries.

Both unsupported questions — the company mascot's favourite colour and the cricket World Cup winner — correctly returned the fallback: `I could not find that information in the policy documents.`

### Refund-agent tools and rules

The refund agent reuses policy retrieval and reads order information from `data/instamart.db`. The seven scenario definitions are loaded from the operational database.

Tools:

- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

Implemented rules:

- Only completed orders are eligible for a refund.
- A completed order delivered in more than 45 minutes receives a 100% refund of its order amount.
- A completed order delivered in 45 minutes or less receives no late-delivery refund.
- Refunds above INR 1,000 are escalated to a human and are not issued automatically.
- The scenarios assume refund requests are raised within the policy's 7-day request window.

### Seven-scenario evaluation

| Scenario | Scenario outcome | Result |
|---|---|---|
| S1 | Refund INR 128.25 | PASS |
| S2 | Refund INR 135.00 | PASS |
| S3 | Escalate INR 1,999.00 | PASS |
| S4 | Denied — delivery within 45 minutes | PASS |
| S5 | Denied — cancelled order | PASS |
| S6 | Order not found | PASS |
| S7 | `FALLBACK_MAX_ITERATIONS` | PASS |

All seven scenarios produced their expected outcomes.

### Stop-safely and crash/resume demonstrations

| Demonstration | Observed status | Result |
|---|---|---|
| Token limit set to 200 | `FALLBACK_TOKEN_LIMIT` | PASS |
| Time limit set to 0 seconds | `FALLBACK_TIMEOUT` | PASS |
| Repeated tool loop | `FALLBACK_MAX_ITERATIONS` | PASS |
| Crash after the second tool | Checkpoint saved | PASS |
| Resume from checkpoint | Completed refund workflow | PASS |

The crash/resume run executed these tools in order across the two runs:

```text
get_order
search_policy
calculate_refund
issue_refund
```

The resumed workflow continued from saved state rather than repeating completed tools.

### Refund ledger and execution logs

The simulated refund ledger contains two issued-refund records:

| Order ID | Refund amount |
|---|---:|
| `ORD-100048` | INR 128.25 |
| `ORD-100067` | INR 135.00 |

Duplicate refund order IDs: **0**.

The agent execution log contains 10 records. Checkpoint files are stored in `state/`.

Required outputs:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/`

Refund issuance and human escalation are simulated. No real payment gateway or support-ticketing system is called.

### ReAct versus Plan-and-Execute

The implemented refund agent follows a ReAct-style loop: the model observes tool results and selects the next action. This allows the workflow to respond to database and policy observations, such as a missing order, a cancelled order, an eligible refund, or a refund requiring escalation.

A Plan-and-Execute approach would create a plan first and then execute its steps. That can reduce repeated planning for predictable workflows, but unexpected tool results may require replanning.

The S1 ReAct execution used 960 mock tokens across four tool executions. A separate Plan-and-Execute implementation was not benchmarked, so no numerical token comparison is claimed. ReAct was retained because the workflow needs to react to actual tool observations.

## Limitations and Next Steps

- The datasets are synthetic and do not represent real customer behaviour, real transactions, or official Instamart policies.
- The offline mock LLM is deterministic. Its measured accuracy does not establish real-world LLM performance.
- The RAG assistant uses TF-IDF and lexical relevance checks. It can miss semantically related passages and does not provide a general semantic-entailment guarantee.
- Chunk overlap improves the observed fact-hit result, but it does not guarantee that every future policy question will be answered correctly.
- The token-budget guard checks before a model call; the final response can exceed the configured budget slightly.
- The retention list uses a Premium segment label derived from the full-window RFM snapshot, including September. It should not be interpreted as a leakage-free historical ranking.
- Refund issuance and human escalation are simulations; the project does not integrate with a real payment gateway or support platform.
- Production use would require authentication, role-based access control, secure audit logs, stronger idempotency and transaction controls, monitoring, real payment-provider integration, and a human approval workflow.
- Before deployment, the pipeline should be tested on representative real-world data, with temporal validation, model calibration, drift monitoring, and operational review.