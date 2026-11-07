import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pandas as pd

# ==================================================
# PART B - TASK 1: RFM CUSTOMER ANALYSIS
# ==================================================

DATA_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

# Load cleaned datasets
orders = pd.read_csv(DATA_DIR / "orders.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
# Parse timestamps
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Use completed orders only
completed = orders[
    orders["status"].str.lower().eq("completed")
].copy()

# Calculate RFM metrics
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

# Recency = days between snapshot and last completed order
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_ts"]
).dt.days

# Remove temporary timestamp column
rfm = rfm.drop(columns=["last_order_ts"])

# Keep a consistent column order
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Validate RFM data
assert len(rfm) == 1175, f"Expected 1175 customers, got {len(rfm)}"
assert rfm["customer_id"].is_unique
assert rfm["recency"].notna().all()
assert rfm["frequency"].gt(0).all()
assert rfm["monetary"].notna().all()

# Save RFM table
output_path = OUTPUT_DIR / "rfm_table.csv"
rfm.to_csv(output_path, index=False)

# Display reference statistics
print("=" * 60)
print("PART B - TASK 1: RFM TABLE")
print("=" * 60)

print("Total customers:", len(rfm))

print("\nRECENCY")
print("Median:", rfm["recency"].median())
print("Minimum:", rfm["recency"].min())
print("Maximum:", rfm["recency"].max())

print("\nFREQUENCY")
print("Mean:", round(rfm["frequency"].mean(), 2))
print("Maximum:", rfm["frequency"].max())

print("\nMONETARY")
print("Mean INR:", round(rfm["monetary"].mean(), 2))
print("Maximum INR:", round(rfm["monetary"].max(), 2))

print("\nSaved:", output_path)
print("ALL RFM CHECKS PASSED")
# ============================================================
# TASK 2: K-MEANS CLUSTERING EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 2: K-MEANS CLUSTERING")
print("=" * 60)

features = ["recency", "frequency", "monetary"]

X = rfm[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })

    print(
        f"K={k} | "
        f"Inertia={inertia:.1f} | "
        f"Silhouette={silhouette:.3f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "kmeans_evaluation.csv",
    index=False
)

print("\nSaved: outputs/kmeans_evaluation.csv")
print("K-MEANS EVALUATION COMPLETED")
# ============================================================
# TASK 3: CUSTOMER SEGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 3: CUSTOMER SEGMENTATION")
print("=" * 60)

FINAL_K = 4

final_kmeans = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(X_scaled)

profile = (
    rfm.groupby("segment_id")
    .agg(
        customers=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    )
    .round(2)
)

print("\nSEGMENT PROFILES")
print(profile.to_string())

# Premium segment = highest mean monetary value
premium_id = profile["mean_monetary"].idxmax()

rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

# Business names based on segment characteristics
# Names are assigned after inspecting the profiles.
segment_names = {
    0: "Occasional Customers",
    1: "Premium High-Value Customers",
    2: "Loyal Frequent Customers",
    3: "At-Risk Dormant Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Save final customer segmentation output
segments_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
].copy()

segments_output.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

assert len(segments_output) == 1175
assert segments_output["segment_name"].notna().all()
assert segments_output["is_premium"].sum() == 138
assert segments_output.loc[
    segments_output["is_premium"], "segment_id"
].eq(1).all()

print("\nSaved: outputs/customer_segments.csv")
print("Premium customers:", int(segments_output["is_premium"].sum()))
print("ALL TASK 3 CHECKS PASSED")

# PCA for dimensionality reduction
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA EXPLAINED VARIANCE")
print("PC1:", round(pca.explained_variance_ratio_[0], 3))
print("PC2:", round(pca.explained_variance_ratio_[1], 3))
print("Total:", round(sum(pca.explained_variance_ratio_), 3))

print("\nPremium segment ID:", premium_id)

print("\nTASK 3 PROFILE CHECK COMPLETED")
# ============================================================
# TASK 4: LAPSED CUSTOMER COHORT
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 4: LAPSED CUSTOMER COHORT")
print("=" * 60)

CUTOFF_DATE = pd.Timestamp("2025-09-01")
PERIOD_END = pd.Timestamp("2025-10-01")

# Completed orders only
completed_orders = orders[
    orders["status"].str.lower().eq("completed")
].copy()

# Orders before cutoff define cohort eligibility
pre_cutoff = completed_orders[
    completed_orders["order_ts"] < CUTOFF_DATE
].copy()

# Customers with at least one completed order before cutoff
cohort_ids = pre_cutoff["customer_id"].unique()

# Completed orders during September
september_orders = completed_orders[
    (completed_orders["order_ts"] >= CUTOFF_DATE)
    & (completed_orders["order_ts"] < PERIOD_END)
]

active_september_ids = set(september_orders["customer_id"])

cohort = pd.DataFrame({
    "customer_id": cohort_ids
})

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_september_ids)
).astype(int)

print("Cohort customers:", len(cohort))
print("Lapsed customers:", int(cohort["lapsed"].sum()))
print("Not lapsed:", int((cohort["lapsed"] == 0).sum()))
print(
    "Lapsed rate:",
    round(cohort["lapsed"].mean() * 100, 2),
    "%"
)

assert len(cohort) == 1156
assert cohort["lapsed"].sum() == 443

cohort.to_csv(
    OUTPUT_DIR / "lapse_cohort.csv",
    index=False
)

print("\nSaved: outputs/lapse_cohort.csv")
print("ALL TASK 4 COHORT CHECKS PASSED")
# ============================================================
# TASK 5: LAPSED CUSTOMER FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 5: FEATURE ENGINEERING")
print("=" * 60)

# Only orders strictly before the cutoff are model features
all_pre = orders[orders["order_ts"] < CUTOFF_DATE].copy()

completed_pre = all_pre[
    all_pre["status"].str.lower().eq("completed")
].copy()

cancelled_pre = all_pre[
    all_pre["status"].str.lower().eq("cancelled")
].copy()

# Completed-order features
completed_features = (
    completed_pre.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    )
    .reset_index()
)

completed_features["recency"] = (
    CUTOFF_DATE - completed_features["last_order_ts"]
).dt.days

completed_features = completed_features.drop(
    columns=["last_order_ts"]
)

# Cancellation rate: cancelled / all orders before cutoff
total_orders = (
    all_pre.groupby("customer_id")["order_id"]
    .nunique()
    .rename("total_orders")
)

cancelled_counts = (
    cancelled_pre.groupby("customer_id")["order_id"]
    .nunique()
    .rename("cancelled_orders")
)

cancellation_features = pd.concat(
    [total_orders, cancelled_counts],
    axis=1
).fillna(0)

cancellation_features["cancellation_rate"] = (
    cancellation_features["cancelled_orders"]
    / cancellation_features["total_orders"]
)

# Most frequently ordered product category
category_data = completed_pre.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favorite_category = (
    category_data.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favorite_category"})
)

# Customer demographic and tenure features
customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_features["signup_date"] = pd.to_datetime(
    customer_features["signup_date"]
)

customer_features["tenure_days"] = (
    CUTOFF_DATE - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(
    columns=["signup_date"]
)

# Combine all features with cohort labels
feature_data = (
    cohort
    .merge(completed_features, on="customer_id", how="left")
    .merge(
        cancellation_features[["cancellation_rate"]],
        left_on="customer_id",
        right_index=True,
        how="left"
    )
    .merge(favorite_category, on="customer_id", how="left")
    .merge(customer_features, on="customer_id", how="left")
)

# Check for missing feature values before preprocessing
feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favorite_category",
    "city",
    "age",
    "gender",
    "tenure_days"
]

print("\nFEATURE MISSING VALUES")
print(feature_data[feature_columns].isna().sum().to_string())

print("\nFeature dataset shape:", feature_data.shape)
print("Cohort customers:", len(feature_data))

# Save feature dataset
feature_data.to_csv(
    OUTPUT_DIR / "lapse_features.csv",
    index=False
)

print("\nSaved: outputs/lapse_features.csv")
print("TASK 5 FEATURE ENGINEERING COMPLETED")
# ============================================================
# TASK 6: MODEL TRAINING, TUNING AND EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 6: MODEL TRAINING AND EVALUATION")
print("=" * 60)

# Separate features and target
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favorite_category",
    "city",
    "gender"
]

X = feature_data[numeric_features + categorical_features].copy()
y = feature_data["lapsed"].copy()

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nDATA SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training lapsed:", int(y_train.sum()))
print("Testing lapsed:", int(y_test.sum()))

assert len(X_train) == 924
assert len(X_test) == 232

# Baseline: always predict not lapsed
baseline_predictions = [0] * len(y_test)

print("\nBASELINE MODEL")
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Lapsed-class F1:", round(
    f1_score(y_test, baseline_predictions, zero_division=0), 4
))

# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

# Combined preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_parameters = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

lr_search = GridSearchCV(
    lr_pipeline,
    lr_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTRAINING LOGISTIC REGRESSION...")
lr_search.fit(X_train, y_train)

lr_predictions = lr_search.predict(X_test)

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_parameters = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("TRAINING RANDOM FOREST...")
rf_search.fit(X_train, y_train)

rf_predictions = rf_search.predict(X_test)

# Model evaluation
def evaluate_model(name, y_true, predictions):
    print("\n" + "-" * 45)
    print(name)
    print("-" * 45)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    print("Accuracy:", round(
        accuracy_score(y_true, predictions), 4
    ))

    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 4
    ))

    print("Recall:", round(
        recall_score(y_true, predictions, zero_division=0), 4
    ))

    print("F1 Score:", round(
        f1_score(y_true, predictions, zero_division=0), 4
    ))


# Outside the function — zero indentation
evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    lr_predictions
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions
)

print("\nBEST PARAMETERS")
print("Logistic Regression:", lr_search.best_params_)
print("Random Forest:", rf_search.best_params_)

print("\nBEST CROSS-VALIDATION F1")
print("Logistic Regression:", round(lr_search.best_score_, 4))
print("Random Forest:", round(rf_search.best_score_, 4))

print("\nTASK 6 MODEL TRAINING COMPLETED")
# ============================================================
# TASK 7: CLASS BALANCING EXPERIMENT
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 7: CLASS BALANCING")
print("=" * 60)

# Select best Logistic Regression model
best_lr = lr_search.best_estimator_

# Get original training class counts
class_counts = y_train.value_counts()

print("\nORIGINAL TRAINING CLASS COUNTS")
print(class_counts.to_dict())

# Separate training rows by class
X_majority = X_train[y_train == 0]
X_minority = X_train[y_train == 1]

# Oversample lapsed customers with replacement
X_minority_balanced = X_minority.sample(
    n=len(X_majority),
    replace=True,
    random_state=42
)

# Combine original majority and balanced minority rows
X_balanced = pd.concat(
    [X_majority, X_minority_balanced]
)

y_balanced = pd.Series(
    [0] * len(X_majority) + [1] * len(X_minority_balanced),
    index=X_balanced.index,
    name="lapsed"
)

# Shuffle balanced training data
balanced_data = X_balanced.copy()
balanced_data["lapsed"] = y_balanced

balanced_data = balanced_data.sample(
    frac=1,
    random_state=42
)

X_balanced = balanced_data.drop(columns=["lapsed"])
y_balanced = balanced_data["lapsed"]

print("\nBALANCED TRAINING CLASS COUNTS")
print(y_balanced.value_counts().to_dict())

# Clone the selected model pipeline
from sklearn.base import clone

balanced_lr = clone(best_lr)

# Train on balanced training data only
balanced_lr.fit(X_balanced, y_balanced)

# Evaluate on original untouched test set
balanced_predictions = balanced_lr.predict(X_test)

print("\nCLASS BALANCING COMPARISON")

original_precision = precision_score(
    y_test, lr_predictions, zero_division=0
)
original_recall = recall_score(
    y_test, lr_predictions, zero_division=0
)
original_f1 = f1_score(
    y_test, lr_predictions, zero_division=0
)

balanced_precision = precision_score(
    y_test, balanced_predictions, zero_division=0
)
balanced_recall = recall_score(
    y_test, balanced_predictions, zero_division=0
)
balanced_f1 = f1_score(
    y_test, balanced_predictions, zero_division=0
)

comparison = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1"],
    "Original": [
        original_precision,
        original_recall,
        original_f1
    ],
    "Balanced": [
        balanced_precision,
        balanced_recall,
        balanced_f1
    ]
})

print(comparison.round(4).to_string(index=False))

print("\nBALANCED CONFUSION MATRIX")
print(confusion_matrix(y_test, balanced_predictions))

print("\nTASK 7 CLASS BALANCING COMPLETED")
# ============================================================
# TASK 8: INTERPRETATION, MODEL PERSISTENCE AND RETENTION LIST
# ============================================================

print("\n" + "=" * 60)
print("PART B - TASK 8: INTERPRETATION AND PERSISTENCE")
print("=" * 60)

# ------------------------------------------------------------
# 1. Random Forest feature importance
# ------------------------------------------------------------

best_rf = rf_search.best_estimator_

rf_model = best_rf.named_steps["model"]
rf_preprocessor = best_rf.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTOP 5 RANDOM FOREST FEATURE IMPORTANCES")
print(importance_df.head(5).to_string(index=False))

importance_df.head(5).to_csv(
    OUTPUT_DIR / "top5_feature_importances.csv",
    index=False
)

# ------------------------------------------------------------
# 2. Logistic Regression coefficient interpretation
# ------------------------------------------------------------

lr_model = lr_search.best_estimator_.named_steps["model"]
lr_preprocessor = lr_search.best_estimator_.named_steps["preprocessor"]

lr_feature_names = lr_preprocessor.get_feature_names_out()
coefficients = lr_model.coef_[0]

coefficient_df = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": coefficients
})

coefficient_df["abs_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

top_coefficients = coefficient_df.sort_values(
    "abs_coefficient",
    ascending=False
).head(3)

print("\nTOP 3 LOGISTIC REGRESSION COEFFICIENTS")
print(top_coefficients[
    ["feature", "coefficient"]
].to_string(index=False))

# ------------------------------------------------------------
# 3. Save and reload selected Logistic Regression model
# ------------------------------------------------------------

model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(lr_search.best_estimator_, model_path)

loaded_model = joblib.load(model_path)

original_predictions = lr_search.best_estimator_.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

assert (original_predictions == reloaded_predictions).all()

print("\nMODEL PERSISTENCE")
print("Saved model:", model_path)
print("Prediction reproduction:", "PASSED")

# ------------------------------------------------------------
# 4. Premium customer retention list
# ------------------------------------------------------------

# Premium status comes from whole-window RFM segmentation.
# This list is illustrative, not a leakage-free evaluation.

test_results = pd.DataFrame({
    "customer_id": feature_data.loc[
        X_test.index, "customer_id"
    ].values,
    "actually_lapsed": y_test.values,
    "lapse_probability": loaded_model.predict_proba(X_test)[:, 1]
})

premium_test_ids = set(
    segments_output.loc[
        segments_output["is_premium"],
        "customer_id"
    ]
)

retention_list = test_results[
    test_results["customer_id"].isin(premium_test_ids)
].copy()

retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention_list = retention_list[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention_list.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False
)

print("\nTOP PREMIUM CUSTOMERS FOR RETENTION")
print(retention_list.to_string(index=False))

print("\nRetention list size:", len(retention_list))
print(
    "Actually lapsed in retention list:",
    int(retention_list["actually_lapsed"].sum())
)

print("\nSaved: outputs/retention_list.csv")
print("TASK 8 COMPLETED")