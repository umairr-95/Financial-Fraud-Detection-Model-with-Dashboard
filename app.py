from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# Page setup and paths
# ------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_fraud_model.joblib"
COMPARISON_PATH = PROJECT_DIR / "models" / "model_comparison.csv"

DATA_CANDIDATES = [
    PROJECT_DIR / "Raw_dataset" / "financial_fraud_detection_feature_engineered.csv",
    PROJECT_DIR / "data" / "financial_fraud_detection_feature_engineered.csv",
]

DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)

TARGET = "Fraudulent"
ID_AND_RAW_COLUMNS = ["Transaction_ID", "Customer_ID", "Transaction_Date", TARGET]


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def money(value):
    return f"{value:,.2f}"


def get_model_features(model):
    """Return original input feature names recorded by the fitted pipeline."""
    try:
        return list(model.named_steps["preprocessor"].feature_names_in_)
    except Exception:
        return []


def prepare_model_frame(frame, model):
    """Align data to the original model input columns and preserve their order."""
    expected = get_model_features(model)
    if not expected:
        raise ValueError("Could not determine the trained model's input columns.")
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(
            "The data is missing model input columns: " + ", ".join(missing)
        )
    return frame[expected].copy()


def feature_importance_frame(model):
    """Extract tree or linear-model importance from the fitted pipeline."""
    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"Feature {i+1}" for i in range(0, 0)])

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_).mean(axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    if len(names) != len(values):
        names = np.array([f"Feature {i+1}" for i in range(len(values))])

    result = pd.DataFrame({"Feature": names, "Importance": values})
    # Shorten pipeline prefixes for readability.
    result["Feature"] = result["Feature"].str.replace(
        r"^(num|cat)__", "", regex=True
    )
    return result.sort_values("Importance", ascending=False)


def add_download_button(dataframe, label, filename):
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ------------------------------------------------------------
# Load required assets
# ------------------------------------------------------------
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Navigate to module",
    [
        "Executive Dashboard",
        "Dataset Overview",
        "EDA Dashboard",
        "Feature Engineering",
        "Fraud Intelligence",
        "Model Comparison",
        "Fraud Prediction",
        "Fraud Analytics",
        "Explainable AI",
        "Real-Time Monitoring",
        "Fraud Forecasting",
        "Graph Analytics",
    ],
)

st.title("🛡️ Financial Fraud Detection System")
st.caption("Explore transaction patterns, evaluate the trained model, and investigate potential fraud.")

if DATA_PATH is None:
    st.error(
        "Feature-engineered CSV not found. Place "
        "'financial_fraud_detection_feature_engineered.csv' in either "
        "'Raw_dataset' or 'data' beside app.py."
    )
    st.stop()

df = load_data(str(DATA_PATH))

if TARGET not in df.columns:
    st.error("The dataset must contain a 'Fraudulent' target column.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "Saved model not found at models/best_fraud_model.joblib. "
        "Run ModelTraining.ipynb and save the trained model first."
    )
    st.stop()

model = load_model(str(MODEL_PATH))
model_features = get_model_features(model)

if not model_features:
    st.error("Could not read input feature names from the saved model pipeline.")
    st.stop()

# Common dataset definitions
df = df.copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
labeled = df[df[TARGET].isin([0, 1])].copy()
fraud_count = int((labeled[TARGET] == 1).sum())
legit_count = int((labeled[TARGET] == 0).sum())
total_labeled = len(labeled)
fraud_rate = (fraud_count / total_labeled * 100) if total_labeled else 0.0

numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()

# Parse transaction date where available
date_series = None
if "Transaction_Date" in df.columns:
    date_series = pd.to_datetime(df["Transaction_Date"], errors="coerce")


# ------------------------------------------------------------
# 1. Executive Dashboard
# ------------------------------------------------------------
if page == "Executive Dashboard":
    st.header("1. Executive Dashboard")
    st.write("High-level view of the labeled transactions in the supplied dataset.")

    amount_sum = (
        pd.to_numeric(df["Transaction_Amount"], errors="coerce").sum()
        if "Transaction_Amount" in df.columns else np.nan
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{len(df):,}")
    c2.metric("Fraud cases", f"{fraud_count:,}")
    c3.metric("Legitimate cases", f"{legit_count:,}")
    c4.metric("Fraud rate", f"{fraud_rate:.2f}%")

    if pd.notna(amount_sum):
        st.metric("Total transaction amount", money(amount_sum))

    left, right = st.columns(2)
    with left:
        st.subheader("Transaction class split")
        fig, ax = plt.subplots()
        ax.pie(
            [legit_count, fraud_count],
            labels=["Legitimate", "Fraudulent"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Transaction amount by class")
        if "Transaction_Amount" in df.columns:
            plot_df = df.copy()
            plot_df["Transaction_Amount"] = pd.to_numeric(
                plot_df["Transaction_Amount"], errors="coerce"
            )
            plot_df["Class"] = plot_df[TARGET].map({0: "Legitimate", 1: "Fraudulent"})
            fig, ax = plt.subplots()
            for class_name in ["Legitimate", "Fraudulent"]:
                values = plot_df.loc[
                    plot_df["Class"] == class_name, "Transaction_Amount"
                ].dropna()
                if len(values):
                    ax.hist(values, bins=25, alpha=0.55, label=class_name)
            ax.set_xlabel("Transaction amount")
            ax.set_ylabel("Number of transactions")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Transaction_Amount is not present in this dataset.")

    st.subheader("Recent rows")
    st.dataframe(df.head(10), use_container_width=True)
    add_download_button(df.head(1000), "Download first 1,000 rows", "transaction_sample.csv")


# ------------------------------------------------------------
# 2. Dataset Overview
# ------------------------------------------------------------
elif page == "Dataset Overview":
    st.header("2. Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c4.metric("Cells missing", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Column types and completeness")
    overview = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(overview, use_container_width=True)
    st.subheader("Missing values by column")
    missing = overview.sort_values("Missing values", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(missing["Column"], missing["Missing values"])
    ax.set_ylabel("Missing cells")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    st.subheader("Filter records")
    selected_cols = st.multiselect("Columns to display", list(df.columns), default=list(df.columns[:min(8, len(df.columns))]))
    st.dataframe(df[selected_cols].head(500), use_container_width=True)
    add_download_button(df, "Download complete dataset", "dataset_overview.csv")


# ------------------------------------------------------------
# 3. EDA Dashboard
# ------------------------------------------------------------
elif page == "EDA Dashboard":
    st.header("3. Exploratory Data Analysis")
    numeric_options = [c for c in numeric_columns if c != TARGET]
    categorical_options = [c for c in categorical_columns if c != TARGET]

    if numeric_options:
        selected_numeric = st.selectbox("Choose a numeric feature", numeric_options)
        left, right = st.columns(2)
        with left:
            st.subheader(f"Distribution: {selected_numeric}")
            values = pd.to_numeric(df[selected_numeric], errors="coerce").dropna()
            fig, ax = plt.subplots()
            ax.hist(values, bins=30, edgecolor="black")
            ax.set_xlabel(selected_numeric)
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader(f"{selected_numeric} by class")
            if TARGET in df.columns:
                plot_df = df[[selected_numeric, TARGET]].copy()
                plot_df[selected_numeric] = pd.to_numeric(plot_df[selected_numeric], errors="coerce")
                plot_df["Class"] = plot_df[TARGET].map({0: "Legitimate", 1: "Fraudulent"})
                fig, ax = plt.subplots()
                for name in ["Legitimate", "Fraudulent"]:
                    vals = plot_df.loc[plot_df["Class"] == name, selected_numeric].dropna()
                    if len(vals):
                        ax.hist(vals, bins=25, alpha=0.5, label=name)
                ax.legend()
                ax.set_xlabel(selected_numeric)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                plt.close(fig)

    if categorical_options:
        cat = st.selectbox("Choose a categorical feature", categorical_options)
        top = df[cat].fillna("Missing").astype(str).value_counts().head(15)
        st.subheader(f"Most common values: {cat}")
        fig, ax = plt.subplots()
        ax.barh(top.index[::-1], top.values[::-1])
        ax.set_xlabel("Transaction count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if TARGET in df.columns:
            fraud_by_category = (
                df.assign(_class=df[TARGET].map({0: "Legitimate", 1: "Fraudulent"}))
                .groupby(cat, dropna=False)["_class"]
                .apply(lambda s: (s == "Fraudulent").sum())
                .sort_values(ascending=False)
                .head(15)
            )
            st.subheader(f"Fraud counts by {cat}")
            st.bar_chart(fraud_by_category)

    numeric_for_corr = df.select_dtypes(include="number")
    if numeric_for_corr.shape[1] >= 2:
        st.subheader("Numeric correlation heatmap")
        corr = numeric_for_corr.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=75, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ------------------------------------------------------------
# 4. Feature Engineering
# ------------------------------------------------------------
elif page == "Feature Engineering":
    st.header("4. Feature Engineering")
    st.write("Inspect derived time and transaction-risk features that are already present in the feature-engineered CSV.")
    engineered = [
        "Transaction_Hour", "Day_of_Week", "Transaction_Month",
        "Is_Weekend", "High_Amount", "Is_Night",
    ]
    present = [c for c in engineered if c in df.columns]
    if present:
        st.success(f"Found {len(present)} engineered columns: {', '.join(present)}")
        st.dataframe(df[present].describe().transpose(), use_container_width=True)
        chosen = st.selectbox("Visualize an engineered feature", present)
        counts = df[chosen].fillna("Missing").astype(str).value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel(chosen)
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No expected engineered columns were found. Check the output of FeatureEngineering.ipynb.")

    if "Transaction_Date" in df.columns:
        dates = pd.to_datetime(df["Transaction_Date"], errors="coerce")
        valid = dates.notna()
        if valid.any():
            derived = pd.DataFrame({"Date": dates[valid]})
            derived["Hour"] = derived["Date"].dt.hour
            derived["Day of week"] = derived["Date"].dt.day_name()
            st.subheader("Date-derived values preview")
            st.dataframe(derived.head(20), use_container_width=True)
    st.caption("This page inspects existing features; it does not silently rewrite your training dataset.")


# ------------------------------------------------------------
# 5. Fraud Intelligence
# ------------------------------------------------------------
elif page == "Fraud Intelligence":
    st.header("5. Fraud Intelligence")
    st.write("Explore suspicious patterns in labeled transactions. These are indicators for investigation, not proof of fraud.")
    fraud_df = df[df[TARGET] == 1].copy()
    if fraud_df.empty:
        st.warning("No rows labeled as fraudulent were found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraudulent rows", f"{len(fraud_df):,}")
        if "Transaction_Amount" in fraud_df.columns:
            amounts = pd.to_numeric(fraud_df["Transaction_Amount"], errors="coerce")
            c2.metric("Average fraud amount", money(amounts.mean()) if amounts.notna().any() else "N/A")
            c3.metric("Median fraud amount", money(amounts.median()) if amounts.notna().any() else "N/A")

        intel_features = [
            c for c in ["Merchant_Category", "Payment_Method", "Device_Type",
                        "Location", "Suspicious_Keyword", "Is_International",
                        "Is_Night", "High_Amount"] if c in fraud_df.columns
        ]
        if intel_features:
            selected = st.selectbox("Investigate pattern by", intel_features)
            pattern = fraud_df[selected].fillna("Missing").astype(str).value_counts().head(15)
            fig, ax = plt.subplots()
            ax.barh(pattern.index[::-1], pattern.values[::-1])
            ax.set_xlabel("Fraud-labeled transactions")
            ax.set_ylabel(selected)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            overall = df[selected].fillna("Missing").astype(str).value_counts()
            fraud_counts = fraud_df[selected].fillna("Missing").astype(str).value_counts()
            rate = (fraud_counts / overall.replace(0, np.nan) * 100).dropna().sort_values(ascending=False).head(15)
            st.subheader(f"Fraud rate by {selected} (top categories)")
            st.bar_chart(rate)
        st.subheader("Fraud-labeled records for review")
        st.dataframe(fraud_df.head(200), use_container_width=True)
        add_download_button(fraud_df, "Download fraud-labeled records", "fraud_intelligence_records.csv")


# ------------------------------------------------------------
# 6. Model Comparison
# ------------------------------------------------------------
elif page == "Model Comparison":
    st.header("6. Model Comparison")
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison.round(4), use_container_width=True)
        score_cols = [c for c in ["Precision", "Recall", "F1-Score", "ROC-AUC", "Accuracy"] if c in comparison.columns]
        if score_cols and "Model" in comparison.columns:
            chart = comparison.set_index("Model")[score_cols]
            st.subheader("Evaluation metrics by model")
            st.bar_chart(chart)
            if "F1-Score" in comparison.columns:
                best_row = comparison.loc[comparison["F1-Score"].idxmax()]
                st.info(f"Highest F1-score in the saved report: {best_row['Model']} ({best_row['F1-Score']:.3f}).")
        add_download_button(comparison, "Download model comparison", "model_comparison.csv")
    else:
        st.warning("models/model_comparison.csv was not found. Run the model comparison cell in ModelTraining.ipynb and save the CSV.")
    st.caption("Metrics reflect the evaluation report saved from your notebook; this page does not retrain models.")


# ------------------------------------------------------------
# 7. Fraud Prediction
# ------------------------------------------------------------
elif page == "Fraud Prediction":
    st.header("7. Fraud Prediction")
    st.write("Enter transaction details to get a prediction from the saved model. Predictions are decision support, not a guarantee.")
    st.caption("Fields are generated from the features expected by your saved model.")

    defaults = df
    with st.form("prediction_form"):
        values = {}
        form_cols = st.columns(2)
        for i, feature in enumerate(model_features):
            col = form_cols[i % 2]
            with col:
                if feature not in df.columns:
                    st.warning(f"Feature '{feature}' is missing from the dataset; using an empty default.")
                    values[feature] = None
                    continue
                series = df[feature]
                if pd.api.types.is_numeric_dtype(series):
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    default_val = float(numeric_series.median()) if len(numeric_series) else 0.0
                    if pd.api.types.is_integer_dtype(series):
                        values[feature] = st.number_input(feature, value=int(round(default_val)))
                    else:
                        values[feature] = st.number_input(feature, value=float(default_val), format="%.4f")
                else:
                    choices = series.dropna().astype(str).value_counts().head(100).index.tolist()
                    if not choices:
                        choices = ["Unknown"]
                    values[feature] = st.selectbox(feature, choices)

        submitted = st.form_submit_button("Predict fraud risk")

    if submitted:
        input_df = pd.DataFrame([values], columns=model_features)
        try:
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0, 1])
            left, right = st.columns(2)
            left.metric("Predicted class", "Fraudulent" if prediction == 1 else "Legitimate")
            right.metric("Estimated fraud probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
            if prediction == 1:
                st.error("The model flags this transaction as potentially fraudulent. Review it before taking action.")
            else:
                st.success("The model classifies this transaction as legitimate; this does not eliminate all risk.")
        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")

    st.subheader("Batch prediction from CSV")
    uploaded = st.file_uploader("Upload a CSV with the model's input columns", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            model_input = prepare_model_frame(batch, model)
            batch["Predicted_Fraudulent"] = model.predict(model_input)
            batch["Fraud_Probability"] = model.predict_proba(model_input)[:, 1]
            st.dataframe(batch.head(100), use_container_width=True)
            add_download_button(batch, "Download predictions", "fraud_predictions.csv")
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


# ------------------------------------------------------------
# 8. Fraud Analytics
# ------------------------------------------------------------
elif page == "Fraud Analytics":
    st.header("8. Fraud Analytics")
    st.write("Summaries of fraud counts, rates, and transaction values.")
    fraud_only = df[df[TARGET] == 1]
    legit_only = df[df[TARGET] == 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud rate", f"{fraud_rate:.2f}%")
    if "Transaction_Amount" in df.columns:
        amounts = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        c2.metric("Overall average amount", money(amounts.mean()) if amounts.notna().any() else "N/A")
        fraud_amounts = pd.to_numeric(fraud_only["Transaction_Amount"], errors="coerce") if "Transaction_Amount" in fraud_only else pd.Series(dtype=float)
        c3.metric("Fraud amount total", money(fraud_amounts.sum()) if fraud_amounts.notna().any() else "N/A")

    group_choices = [c for c in ["Merchant_Category", "Payment_Method", "Device_Type", "Location", "Is_International"] if c in df.columns]
    if group_choices:
        group_col = st.selectbox("Group analytics by", group_choices)
        summary = df.groupby(group_col, dropna=False).agg(
            Transactions=(TARGET, "size"),
            Fraud_cases=(TARGET, lambda s: (s == 1).sum()),
        )
        summary["Fraud_rate_%"] = (summary["Fraud_cases"] / summary["Transactions"] * 100).round(2)
        summary = summary.sort_values("Fraud_rate_%", ascending=False)
        st.subheader(f"Fraud summary by {group_col}")
        st.dataframe(summary, use_container_width=True)
        st.bar_chart(summary[["Transactions", "Fraud_cases"]])
        st.subheader("Fraud rate (%)")
        st.bar_chart(summary["Fraud_rate_%"])
        add_download_button(summary.reset_index(), "Download analytics summary", "fraud_analytics_summary.csv")

    if date_series is not None and date_series.notna().any():
        temporal = df.loc[date_series.notna(), [TARGET]].copy()
        temporal["Date"] = date_series[date_series.notna()].dt.date
        daily = temporal.groupby("Date").agg(
            Transactions=(TARGET, "size"),
            Fraud_cases=(TARGET, lambda s: (s == 1).sum())
        )
        daily["Fraud_rate_%"] = daily["Fraud_cases"] / daily["Transactions"] * 100
        st.subheader("Daily transaction and fraud trends")
        st.line_chart(daily[["Transactions", "Fraud_cases"]])
        st.line_chart(daily[["Fraud_rate_%"]])


# ------------------------------------------------------------
# 9. Explainable AI
# ------------------------------------------------------------
elif page == "Explainable AI":
    st.header("9. Explainable AI")
    st.write("Show which transformed features the fitted model relies on most. Importance is not proof of causation.")
    importance = feature_importance_frame(model)
    if not importance.empty:
        top_n = st.slider("Number of features to display", 5, min(30, len(importance)), min(15, len(importance)))
        top = importance.head(top_n).sort_values("Importance")
        st.subheader("Top model feature importances")
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.28)))
        ax.barh(top["Feature"], top["Importance"])
        ax.set_xlabel("Importance (or absolute coefficient)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(importance.head(top_n), use_container_width=True)
        add_download_button(importance, "Download feature importance", "feature_importance.csv")
    else:
        st.warning("This estimator does not expose built-in feature importance or coefficients. Consider permutation importance in a separate analysis.")
    st.info("Categorical one-hot encoded values may appear as separate feature names. For a reliable explanation of an individual prediction, consider adding SHAP after validating compatibility with the selected model.")


# ------------------------------------------------------------
# 10. Real-Time Monitoring
# ------------------------------------------------------------
elif page == "Real-Time Monitoring":
    st.header("10. Real-Time Monitoring")
    st.write("This page demonstrates monitoring on the latest records in your CSV. It is not a live bank feed.")
    st.warning("Demo mode: records are read from the dataset. A live feed requires an API, stream, or continuously updated source.")
    n = st.slider("Number of records to monitor", 10, min(500, max(10, len(df))), min(100, max(10, len(df))))
    try:
        batch = prepare_model_frame(df.tail(n), model)
        monitored = df.tail(n).copy()
        monitored["Model_Prediction"] = model.predict(batch)
        monitored["Fraud_Probability"] = model.predict_proba(batch)[:, 1]
        monitored["Risk_Level"] = pd.cut(
            monitored["Fraud_Probability"],
            bins=[-0.001, 0.30, 0.70, 1.001],
            labels=["Low", "Medium", "High"]
        ).astype(str)
        c1, c2, c3 = st.columns(3)
        c1.metric("Records scanned", f"{len(monitored):,}")
        c2.metric("Flagged by model", f"{int((monitored['Model_Prediction'] == 1).sum()):,}")
        c3.metric("High risk", f"{int((monitored['Risk_Level'] == 'High').sum()):,}")
        st.subheader("Fraud probability for monitored records")
        plot_data = monitored.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_data.index, plot_data["Fraud_Probability"], marker=".", linewidth=1)
        ax.axhline(0.7, linestyle="--", label="High-risk guide (0.70)")
        ax.set_xlabel("Record index in selected batch")
        ax.set_ylabel("Fraud probability")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Flagged records")
        flagged = monitored[monitored["Model_Prediction"] == 1].sort_values("Fraud_Probability", ascending=False)
        st.dataframe(flagged.head(200), use_container_width=True)
        add_download_button(monitored, "Download monitored batch", "monitoring_batch.csv")
    except Exception as exc:
        st.error(f"Could not score monitoring records: {exc}")


# ------------------------------------------------------------
# 11. Fraud Forecasting
# ------------------------------------------------------------
elif page == "Fraud Forecasting":
    st.header("11. Fraud Forecasting")
    st.write("Summarize historical fraud counts over time. This is a simple baseline trend, not a validated predictive forecast.")
    if date_series is None or not date_series.notna().any():
        st.warning("Transaction_Date is missing or could not be parsed, so time-series analysis is unavailable.")
    else:
        temp = df.loc[date_series.notna(), [TARGET]].copy()
        temp["Date"] = date_series[date_series.notna()].values
        temp = temp.set_index("Date").sort_index()
        frequency = st.selectbox("Time aggregation", ["Daily", "Weekly", "Monthly"], index=1)
        rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[frequency]
        timeline = temp.resample(rule).agg(
            Transactions=(TARGET, "size"),
            Fraud_cases=(TARGET, lambda s: (s == 1).sum())
        )
        timeline["Fraud_rate_%"] = np.where(
            timeline["Transactions"] > 0,
            timeline["Fraud_cases"] / timeline["Transactions"] * 100,
            np.nan
        )
        st.subheader("Historical counts")
        st.line_chart(timeline[["Transactions", "Fraud_cases"]])
        st.subheader("Historical fraud rate")
        st.line_chart(timeline[["Fraud_rate_%"]])

        nonempty = timeline.dropna(subset=["Fraud_cases"])
        if len(nonempty) >= 3:
            st.subheader("Naive next-period baseline")
            recent = nonempty["Fraud_cases"].tail(3)
            baseline = float(recent.mean())
            st.metric("Next period baseline (mean of last 3 periods)", f"{baseline:.1f} fraud cases")
            st.caption("This is a simple rolling-average baseline, not a trained forecasting model. Use more history and out-of-time validation before relying on forecasts.")
        else:
            st.info("There are too few time periods for a useful baseline.")
        add_download_button(timeline.reset_index(), "Download time-series summary", "fraud_time_series.csv")


# ------------------------------------------------------------
# 12. Graph Analytics
elif page == "Graph Analytics":
    st.header("12. Graph Analytics")
    st.write("Explore connections between entities in the transaction dataset.")
    st.caption(
        "Connections show associations in the available data; they do not prove fraud."
    )

    entity_candidates = [
        c for c in [
            "Customer_ID",
            "Location",
            "Merchant_Category",
            "Payment_Method",
            "Device_Type",
        ]
        if c in df.columns
    ]

    if len(entity_candidates) < 2:
        st.warning(
            "Graph Analytics needs at least two available entity columns. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    else:
        source_col = st.selectbox(
            "First node type",
            entity_candidates,
            index=0,
            key="graph_source_col",
        )
        remaining = [c for c in entity_candidates if c != source_col]
        target_col = st.selectbox(
            "Connected node type",
            remaining,
            index=0,
            key="graph_target_col",
        )

        if source_col == target_col:
            st.warning("Choose two different node types to build the network.")
        else:
            max_rows_available = max(1, min(5000, len(df)))
            default_rows = min(1000, max_rows_available)
            max_rows = st.slider(
                "Transaction rows to analyse",
                min_value=1,
                max_value=max_rows_available,
                value=default_rows,
                key="graph_max_rows",
            )

            graph_columns = [source_col, target_col]
            if TARGET in df.columns:
                graph_columns.append(TARGET)

            graph_df = df[graph_columns].dropna(subset=[source_col, target_col]).head(max_rows).copy()
            graph_df[source_col] = graph_df[source_col].astype(str)
            graph_df[target_col] = graph_df[target_col].astype(str)

            if graph_df.empty:
                st.warning("No usable connections were found for these columns.")
            else:
                if TARGET in graph_df.columns:
                    edge_summary = (
                        graph_df.groupby([source_col, target_col], as_index=False)
                        .agg(
                            Transactions=(source_col, "size"),
                            Fraud_cases=(TARGET, lambda values: int((values == 1).sum())),
                        )
                        .sort_values("Transactions", ascending=False)
                    )
                else:
                    edge_summary = (
                        graph_df.groupby([source_col, target_col], as_index=False)
                        .size()
                        .rename(columns={"size": "Transactions"})
                        .sort_values("Transactions", ascending=False)
                    )
                    edge_summary["Fraud_cases"] = 0

                st.subheader("Connection summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Unique connections", len(edge_summary))
                col2.metric("Source entities", graph_df[source_col].nunique())
                col3.metric("Connected entities", graph_df[target_col].nunique())

                st.subheader("Most frequent connections")
                st.dataframe(edge_summary.head(30), use_container_width=True)

                st.subheader("Connection network")
                try:
                    import networkx as nx

                    max_edges = st.slider(
                        "Connections to display",
                        min_value=5,
                        max_value=max(5, min(60, len(edge_summary))),
                        value=min(20, max(5, len(edge_summary))),
                        key="graph_max_edges",
                    )

                    display_edges = edge_summary.head(max_edges)
                    G = nx.Graph()

                    for _, row in display_edges.iterrows():
                        source_node = f"{source_col}: {row[source_col]}"
                        target_node = f"{target_col}: {row[target_col]}"
                        G.add_edge(
                            source_node,
                            target_node,
                            weight=int(row["Transactions"]),
                            fraud=int(row["Fraud_cases"]),
                        )

                    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                        st.info("There are not enough connections to draw a network.")
                    else:
                        fig, ax = plt.subplots(figsize=(14, 9))
                        pos = nx.spring_layout(
                            G,
                            k=1.4 / max(1, G.number_of_nodes() ** 0.5) * 5,
                            iterations=150,
                            seed=42,
                            weight="weight",
                        )

                        edge_widths = [
                            0.7 + min(2.5, G[u][v]["weight"] * 0.25)
                            for u, v in G.edges()
                        ]
                        nx.draw_networkx_edges(
                            G,
                            pos,
                            ax=ax,
                            edge_color="#94A3B8",
                            alpha=0.45,
                            width=edge_widths,
                        )
                        nx.draw_networkx_nodes(
                            G,
                            pos,
                            ax=ax,
                            node_size=420,
                            node_color="#3B82F6",
                            edgecolors="white",
                            linewidths=0.8,
                        )

                        # Keep the chart readable: show labels only on smaller networks.
                        if G.number_of_nodes() <= 25:
                            nx.draw_networkx_labels(
                                G,
                                pos,
                                ax=ax,
                                font_size=7,
                                font_color="#111827",
                            )

                        ax.set_title(
                            f"{source_col} ↔ {target_col} connection network",
                            pad=14,
                        )
                        ax.axis("off")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        if G.number_of_nodes() > 25:
                            st.caption(
                                "Labels are hidden because the network is large. "
                                "Reduce the number of connections to inspect a smaller network."
                            )

                except ImportError:
                    st.info(
                        "Install NetworkX to display the connection network: "
                        "python -m pip install networkx"
                    )

                add_download_button(
                    edge_summary,
                    "Download graph connection summary",
                    "graph_edges.csv",
                )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {DATA_PATH.name}")
st.sidebar.caption("Prototype for analysis and decision support; review flagged cases before action.")
