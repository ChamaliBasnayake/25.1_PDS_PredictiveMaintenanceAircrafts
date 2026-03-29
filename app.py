import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_page_config(
    page_title="Aircraft Predictive Maintenance Dashboard",
    page_icon="✈️",
    layout="wide"
)

PREDICTIONS_PATH = "reports/model_predictions_FD003.csv"
PERFORMANCE_SUMMARY_PATH = "reports/performance_summary.csv"

RUL_MODEL_PATH = "models/xgboost_rul_model_FD003.joblib"
CLASSIFIER_MODEL_PATH = "models/xgboost_classifier_FD003.joblib"
KMEANS_MODEL_PATH = "models/kmeans_model_FD003.pkl"
KMEANS_SCALER_PATH = "models/kmeans_scaler_FD003.pkl"

RUL_FEATURES_PATH = "models/rul_feature_columns.json"
CLASSIFIER_FEATURES_PATH = "models/classifier_feature_columns.json"

REQUIRED_RAW_COLUMNS = (
    ["engine_id", "cycle", "op_set_1", "op_set_2", "op_set_3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)

FIXED_PREDICTIVE_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]


@st.cache_data
def load_predictions():
    if os.path.exists(PREDICTIONS_PATH):
        return pd.read_csv(PREDICTIONS_PATH)
    return None


@st.cache_data
def load_performance_summary():
    if os.path.exists(PERFORMANCE_SUMMARY_PATH):
        return pd.read_csv(PERFORMANCE_SUMMARY_PATH)
    return None


@st.cache_resource
def load_models_and_metadata():
    missing = []

    for path in [
        RUL_MODEL_PATH,
        CLASSIFIER_MODEL_PATH,
        KMEANS_MODEL_PATH,
        KMEANS_SCALER_PATH,
        RUL_FEATURES_PATH,
        CLASSIFIER_FEATURES_PATH,
    ]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        return None, None, None, None, None, None, missing

    rul_model = joblib.load(RUL_MODEL_PATH)
    classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)

    with open(RUL_FEATURES_PATH, "r", encoding="utf-8") as f:
        rul_features = json.load(f)

    with open(CLASSIFIER_FEATURES_PATH, "r", encoding="utf-8") as f:
        classifier_features = json.load(f)

    return (
        rul_model,
        classifier_model,
        kmeans_model,
        kmeans_scaler,
        rul_features,
        classifier_features,
        None
    )


def cluster_name_map(value):
    mapping = {
        0: "Healthy",
        1: "Moderate Degradation",
        2: "High Risk"
    }
    try:
        return mapping.get(int(value), str(value))
    except Exception:
        return str(value)


def validate_raw_input_columns(df: pd.DataFrame):
    return [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]


def create_sample_input_template():
    rows = []
    for engine_id in [1, 2]:
        for cycle in range(1, 6):
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_set_1": 0.0,
                "op_set_2": 0.0,
                "op_set_3": 100.0,
            }
            for i in range(1, 22):
                row[f"sensor_{i}"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def engineer_features_for_inference(raw_df: pd.DataFrame, kmeans_model, kmeans_scaler) -> pd.DataFrame:
    df = raw_df.copy()

    numeric_cols = ["engine_id", "cycle", "op_set_1", "op_set_2", "op_set_3"] + [f"sensor_{i}" for i in range(1, 22)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["engine_id", "cycle"]).copy()
    df["engine_id"] = df["engine_id"].astype(int)
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    predictive_sensors = FIXED_PREDICTIVE_SENSORS
    df_feat = df.copy()

    df_feat["engine_age"] = df_feat["cycle"]
    df_feat["cycle_norm"] = df_feat["cycle"] / df_feat.groupby("engine_id")["cycle"].transform("max")

    rolling_mean = (
        df_feat.groupby("engine_id")[predictive_sensors]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_mean.columns = [f"{c}_mean_5" for c in predictive_sensors]

    rolling_std = (
        df_feat.groupby("engine_id")[predictive_sensors]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    rolling_std.columns = [f"{c}_std_5" for c in predictive_sensors]

    first_cycle_vals = df_feat.groupby("engine_id")[predictive_sensors].transform("first")
    delta_from_start = df_feat[predictive_sensors] - first_cycle_vals
    delta_from_start.columns = [f"{c}_delta_start" for c in predictive_sensors]

    df_feat = pd.concat(
        [df_feat, rolling_mean, rolling_std, delta_from_start],
        axis=1
    ).fillna(0)

    cluster_features_expected = getattr(kmeans_scaler, "feature_names_in_", None)
    if cluster_features_expected is not None:
        cluster_features_expected = list(cluster_features_expected)
    else:
        cluster_features_expected = [f"{c}_mean_5" for c in predictive_sensors]

    X_cluster = df_feat.reindex(columns=cluster_features_expected, fill_value=0)
    X_cluster_scaled = kmeans_scaler.transform(X_cluster)
    df_feat["cluster_label"] = kmeans_model.predict(X_cluster_scaled)

    latest_df = df_feat.groupby("engine_id").tail(1).copy().reset_index(drop=True)
    return latest_df


def build_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "cycle" in out.columns:
        max_cycle = max(float(out["cycle"].max()), 1.0)
        out["normalized_cycle"] = out["cycle"] / max_cycle
    else:
        out["normalized_cycle"] = 0.0

    cluster_weight_map = {0: 0.1, 1: 0.3, 2: 0.6}
    out["cluster_weight"] = out["cluster_label"].map(cluster_weight_map).fillna(0.2)

    out["priority_score"] = (
        0.6 * out["failure_probability"] +
        0.3 * out["normalized_cycle"] +
        0.1 * out["cluster_weight"]
    )

    return out


pred_df = load_predictions()
perf_df = load_performance_summary()

(
    rul_model,
    classifier_model,
    kmeans_model,
    kmeans_scaler,
    rul_features,
    classifier_features,
    missing_assets
) = load_models_and_metadata()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard Overview",
        "Engine Insights",
        "Predict New Engine",
        "Model Performance"
    ]
)

if page == "Dashboard Overview":

    st.title("✈️ Aircraft Predictive Maintenance Dashboard")
    st.subheader("NASA CMAPSS FD003 | Engine Health Monitoring Overview")

    st.markdown("""
This dashboard supports predictive maintenance decision-making using:

• Remaining Useful Life prediction  
• Failure-risk classification (within 30 cycles)  
• Engine health clustering  
• Maintenance priority scoring  
• Prediction for unseen FD003-style engine inputs
""")

    if pred_df is not None and not pred_df.empty:

        total_engines = pred_df["engine_id"].nunique()
        avg_rul = pred_df["predicted_rul"].mean()
        avg_failure_prob = pred_df["failure_probability"].mean()
        max_priority = pred_df["priority_score"].max()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Engines Evaluated", total_engines)
        c2.metric("Average Predicted RUL", f"{avg_rul:.2f}")
        c3.metric("Average Failure Probability", f"{avg_failure_prob:.2f}")
        c4.metric("Highest Priority Score", f"{max_priority:.2f}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Priority Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(pred_df["priority_score"], bins=20)
            ax.set_xlabel("Priority Score")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.subheader("Predicted RUL Distribution")
            fig, ax = plt.subplots()
            ax.hist(pred_df["predicted_rul"], bins=20)
            ax.set_xlabel("Predicted RUL")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Cluster Distribution")
            cluster_counts = pred_df["cluster_label"].value_counts().sort_index()
            cluster_labels = [cluster_name_map(i) for i in cluster_counts.index]

            fig, ax = plt.subplots()
            ax.bar(cluster_labels, cluster_counts.values)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Engine Count")
            st.pyplot(fig)

        with col4:
            st.subheader("Failure Probability vs Predicted RUL")
            fig, ax = plt.subplots()
            ax.scatter(
                pred_df["predicted_rul"],
                pred_df["failure_probability"],
                alpha=0.6
            )
            ax.set_xlabel("Predicted RUL")
            ax.set_ylabel("Failure Probability")
            st.pyplot(fig)

    else:
        st.info("Prediction dataset not found. Upload engines under 'Predict New Engine' to generate results.")

    if missing_assets:
        st.warning("Some model assets are missing:")
        for path in missing_assets:
            st.write(f"- {path}")

elif page == "Engine Insights":
    st.title("🔍 Engine Insights")

    if pred_df is None or pred_df.empty:
        st.error("Prediction file not found or empty.")
    elif "engine_id" not in pred_df.columns:
        st.error("`engine_id` column is missing in predictions file.")
    else:
        display_df = pred_df.copy()

        if "cluster_label" in display_df.columns:
            display_df["cluster_name"] = display_df["cluster_label"].apply(cluster_name_map)

        st.subheader("Top Maintenance Priority Engines")

        top_n = st.slider("Select number of engines to display", 5, 30, 10)
        sort_col = "priority_score" if "priority_score" in display_df.columns else display_df.columns[-1]

        top_df = display_df.sort_values(sort_col, ascending=False).head(top_n)

        cols_to_show = [c for c in [
            "engine_id",
            "predicted_rul",
            "failure_probability",
            "cluster_label",
            "cluster_name",
            "priority_score"
        ] if c in top_df.columns]

        st.dataframe(top_df[cols_to_show], use_container_width=True)

        st.divider()

        st.subheader("Engine Details")

        engine_ids = sorted(display_df["engine_id"].unique().tolist())
        selected_engine = st.selectbox("Select Engine ID", engine_ids)

        row = display_df[display_df["engine_id"] == selected_engine].iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        if "predicted_rul" in row.index:
            c1.metric("Predicted RUL", f"{float(row['predicted_rul']):.2f}")
        if "failure_probability" in row.index:
            c2.metric("Failure Probability", f"{float(row['failure_probability']):.2f}")
        if "cluster_name" in row.index:
            c3.metric("Cluster", row["cluster_name"])
        elif "cluster_label" in row.index:
            c3.metric("Cluster", cluster_name_map(row["cluster_label"]))
        if "priority_score" in row.index:
            c4.metric("Priority Score", f"{float(row['priority_score']):.2f}")

        st.write(row)

elif page == "Predict New Engine":
    st.title("🛠 Predict RUL for New Engine Data")

    if missing_assets:
        st.error("Prediction models are not available. Please make sure all model files exist in the `models/` folder.")
    else:
        st.markdown("""
Upload a CSV file containing **raw FD003-style engine sensor data**.

Required columns:
- `engine_id`
- `cycle`
- `op_set_1`, `op_set_2`, `op_set_3`
- `sensor_1` to `sensor_21`

The app will:
1. engineer features
2. assign cluster labels
3. predict RUL
4. predict failure probability
5. compute maintenance priority score
""")

        st.subheader("Download Input Template")

        template_df = create_sample_input_template()
        template_csv = template_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download sample FD003 input template",
            data=template_csv,
            file_name="fd003_input_template.csv",
            mime="text/csv"
        )

        st.caption("Template preview:")
        st.dataframe(template_df.head(), use_container_width=True)

        st.info(
            "Use multiple rows per engine across cycles. "
            "The app uses the latest cycle per engine after feature engineering."
        )

        uploaded_file = st.file_uploader("Upload raw engine CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview")
                st.dataframe(raw_df.head(), use_container_width=True)

                missing_cols = validate_raw_input_columns(raw_df)
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    pred_input = engineer_features_for_inference(
                        raw_df=raw_df,
                        kmeans_model=kmeans_model,
                        kmeans_scaler=kmeans_scaler
                    )

                    X_reg = pred_input.reindex(columns=rul_features, fill_value=0)
                    X_cls = pred_input.reindex(columns=classifier_features, fill_value=0)

                    pred_input["predicted_rul"] = rul_model.predict(X_reg)
                    pred_input["failure_probability"] = classifier_model.predict_proba(X_cls)[:, 1]

                    pred_input = build_priority_score(pred_input)
                    pred_input["cluster_name"] = pred_input["cluster_label"].apply(cluster_name_map)

                    result_cols = [c for c in [
                        "engine_id",
                        "cycle",
                        "predicted_rul",
                        "failure_probability",
                        "cluster_label",
                        "cluster_name",
                        "priority_score"
                    ] if c in pred_input.columns]

                    st.subheader("Prediction Results")
                    st.dataframe(
                        pred_input[result_cols].sort_values("priority_score", ascending=False),
                        use_container_width=True
                    )

                    csv_out = pred_input[result_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download prediction results as CSV",
                        data=csv_out,
                        file_name="new_engine_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Model Performance":
    st.title("📊 Model Performance")

    if perf_df is not None and not perf_df.empty:
        regression_metrics = perf_df[
            perf_df["metric"].isin(["RMSE", "MAE", "R2", "NASA_score"])
        ].copy()

        classification_metrics = perf_df[
            perf_df["metric"].isin(["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
        ].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Regression Performance")
            st.dataframe(regression_metrics, use_container_width=True)

        with col2:
            st.subheader("Classification Performance")
            st.dataframe(classification_metrics, use_container_width=True)
    else:
        st.info("No performance summary CSV found. Export `summary` from the evaluation notebook to `reports/performance_summary.csv`.")