
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "afterburn_logistic_regression.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "usgs_inventory_missing_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="AfterBurn Sentinel",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styling
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.08);
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. LOAD PROJECT RESOURCES
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_output_csv(filename):
    return pd.read_csv(OUTPUT_DIR / filename)


try:
    model = load_model()
    metadata = load_metadata()
    df = load_dataset()

except Exception as error:
    st.error(f"Unable to load project resources: {error}")
    st.stop()


FEATURES = metadata["features"]
THRESHOLD = float(metadata["threshold"])


FEATURE_LABELS = {
    "StormAccumulation": "Storm Accumulation",
    "StormDuration": "Storm Duration",
    "PeakI60": "Peak Rainfall Intensity (60 min)",
    "RainfallRatio60": "Rainfall Ratio (60 min)",
    "HM23": "HM23",
    "dNBR/1000": "Burn Severity (dNBR/1000)",
    "KF": "KF",
    "UpstreamArea": "Upstream Watershed Area",
    "DaysSinceFire": "Days Since Wildfire",
    "YearAfterFire": "Year After Wildfire"
}


# =====================================================
# 3. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🔥 AfterBurn Sentinel")

st.sidebar.caption(
    "Explainable Post-Wildfire "
    "Debris-Flow Assessment"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Project Overview",
        "Risk Assessment",
        "Explainable AI",
        "Geospatial Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")
st.sidebar.write(metadata["model_name"])
st.sidebar.metric(
    "Classification Threshold",
    f"{THRESHOLD:.2f}"
)

st.sidebar.caption(
    "Research prototype using historical USGS observations."
)


# =====================================================
# 4. PAGE 1: PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    st.title("🔥 AfterBurn Sentinel")

    st.subheader(
        "Explainable Machine Learning Framework "
        "for Post-Wildfire Debris-Flow Assessment"
    )

    st.write("""
    AfterBurn Sentinel is a research prototype that studies
    historical rainfall, soil, burn-related and watershed
    measurements to estimate the likelihood of an observed
    post-wildfire debris-flow response.
    """)

    st.info(
        "This application uses historical observations. "
        "It does not forecast future rainfall or provide "
        "a validated real-time early-warning service."
    )

    st.divider()

    total_observations = len(df)
    total_wildfires = df["FireName"].nunique()
    total_states = df["FireState"].nunique()
    positive_count = int((df["Response"] == 1).sum())
    negative_count = int((df["Response"] == 0).sum())

    positive_percentage = (
        positive_count / total_observations
    ) * 100

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Observations",
        f"{total_observations:,}"
    )

    c2.metric(
        "Wildfire Areas",
        total_wildfires
    )

    c3.metric(
        "States",
        total_states
    )

    c4.metric(
        "Positive Response Rate",
        f"{positive_percentage:.2f}%"
    )

    st.divider()

    left, right = st.columns(2)

   
    with left:

        st.subheader("Historical Response Distribution")

        fig, ax = plt.subplots(figsize=(9, 3.8))

        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        labels = [
            "No Recorded Debris Flow (0)",
            "Recorded Debris Flow (1)"
        ]

        counts = [
            negative_count,
            positive_count
        ]

        bars = ax.barh(
            labels,
            counts,
            color=["#4C9BE8", "#F59E0B"],
            height=0.55
        )

        ax.invert_yaxis()

        ax.set_title(
            "Historical Response Distribution",
            color="white",
            fontsize=14,
            pad=15
        )

        ax.set_xlabel(
            "Number of Observations",
            color="white"
        )

        ax.tick_params(colors="white")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        for bar, count in zip(bars, counts):

            percentage = count / total_observations * 100

            ax.text(
                bar.get_width() + 100,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({percentage:.2f}%)",
                va="center",
                color="white",
                fontsize=10
            )

        ax.set_xlim(0, max(counts) * 1.2)

        for spine in ax.spines.values():
            spine.set_color("#555555")

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


    with right:

        st.subheader("Project Objectives")

        st.markdown("""
        - Study historical post-wildfire observations.
        - Examine rainfall and environmental measurements.
        - Estimate debris-flow response likelihood.
        - Explain predictions using SHAP.
        - Investigate geographical patterns and errors.
        """)

        st.subheader("Selected Model")

        st.write(
            "**Logistic Regression – Reduced Features**"
        )

        st.write(f"Input features: **{len(FEATURES)}**")

        st.write(
            f"Classification threshold: **{THRESHOLD:.2f}**"
        )

    st.divider()

    st.subheader("Technologies Used")

    st.write("""
    Python | Pandas | NumPy | Scikit-learn |
    SHAP | Folium | Streamlit
    """)


# =====================================================
# 5. PAGE 2: RISK ASSESSMENT
# =====================================================

elif page == "Risk Assessment":

    st.title("Debris-Flow Likelihood Assessment")

    st.write(
        "Enter historical storm and environmental "
        "measurements to obtain a model assessment."
    )

    st.info(
        "Enter measurements using the same units and "
        "definitions as the original dataset. Default "
        "values are dataset medians."
    )

    defaults = df[FEATURES].median()

    input_values = {}

    with st.form("risk_assessment_form"):

        st.subheader("Rainfall Measurements")

        rainfall_features = [
            "StormAccumulation",
            "StormDuration",
            "PeakI60",
            "RainfallRatio60"
        ]

        col1, col2 = st.columns(2)

        for index, feature in enumerate(rainfall_features):

            selected_column = col1 if index % 2 == 0 else col2

            with selected_column:

                default_value = float(defaults[feature])

                input_values[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=0.0,
                    value=default_value,
                    step=0.1,
                    format="%.4f",
                    help=(
                        f"Observed dataset range: "
                        f"{df[feature].min():.3f} to "
                        f"{df[feature].max():.3f}"
                    )
                )

        st.divider()

        st.subheader("Environmental and Watershed Measurements")

        environmental_features = [
            "HM23",
            "dNBR/1000",
            "KF",
            "UpstreamArea",
            "DaysSinceFire",
            "YearAfterFire"
        ]

        col3, col4 = st.columns(2)

        for index, feature in enumerate(environmental_features):

            selected_column = col3 if index % 2 == 0 else col4

            with selected_column:

                default_value = float(defaults[feature])

                step_value = (
                    1.0
                    if feature in ["DaysSinceFire", "YearAfterFire"]
                    else 0.01
                )

                input_values[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=0.0,
                    value=default_value,
                    step=step_value,
                    format="%.4f",
                    help=(
                        f"Observed dataset range: "
                        f"{df[feature].min():.3f} to "
                        f"{df[feature].max():.3f}"
                    )
                )

        submitted = st.form_submit_button(
            "Assess Debris-Flow Likelihood",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        st.divider()

        st.subheader("Assessment Results")

        result1, result2 = st.columns(2)

        result1.metric(
            "Estimated Probability of Response = 1",
            f"{probability * 100:.2f}%"
        )

        result2.metric(
            "Classification Threshold",
            f"{THRESHOLD * 100:.0f}%"
        )

        st.progress(probability)

        if prediction == 1:

            st.warning(
                "Positive response pattern detected "
                "(Response = 1)."
            )

        else:

            st.success(
                "Negative response pattern detected "
                "(Response = 0)."
            )

        # Check whether values are outside the central
        # 98% range of observed training data.
        unusual_features = []

        for feature, value in input_values.items():

            lower = df[feature].quantile(0.01)
            upper = df[feature].quantile(0.99)

            if value < lower or value > upper:
                unusual_features.append(
                    FEATURE_LABELS[feature]
                )

        if unusual_features:

            st.warning(
                "Some inputs are outside the central "
                "98% of observed dataset values: "
                + ", ".join(unusual_features)
                + ". Interpret this estimate cautiously."
            )

        with st.expander("View entered measurements"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

        st.caption(
            "This is a historical-data model assessment, "
            "not a validated safety warning. Do not use "
            "it as the sole basis for real-world safety "
            "or evacuation decisions."
        )


# =====================================================
# 6. PAGE 3: EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.title("Explainable Artificial Intelligence")

    st.write(
        "Understand which features influence the model's "
        "predictions using SHAP explanations."
    )

    try:

        importance_df = load_output_csv(
            "shap_global_importance.csv"
        )

        shap_df = load_output_csv(
            "shap_holdout_contributions.csv"
        )

    except Exception as error:

        st.error(
            f"Unable to load SHAP result files: {error}"
        )
        st.stop()

    st.divider()

    
    # Global SHAP Importance - Improved Design

    importance_plot_df = importance_df.copy()

    feature_display_names = {
        "PeakI60": "Peak Rainfall Intensity (60 min)",
        "UpstreamArea": "Upstream Watershed Area",
        "YearAfterFire": "Year After Wildfire",
        "HM23": "HM23",
        "StormDuration": "Storm Duration",
        "RainfallRatio60": "Rainfall Ratio (60 min)",
        "DaysSinceFire": "Days Since Wildfire",
        "dNBR/1000": "Burn Severity (dNBR)",
        "StormAccumulation": "Storm Accumulation",
        "KF": "KF"
    }

    importance_plot_df["Display_Feature"] = (
        importance_plot_df["Feature"]
        .map(feature_display_names)
        .fillna(importance_plot_df["Feature"])
    )

    importance_plot_df = importance_plot_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    bars = ax.barh(
        importance_plot_df["Display_Feature"],
        importance_plot_df["Mean_Absolute_SHAP"],
        color="#4C9BE8",
        height=0.62
    )

    ax.set_title(
        "Global SHAP Feature Importance",
        color="white",
        fontsize=15,
        pad=16
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Value",
        color="white"
    )

    ax.set_ylabel(
        "Environmental and Rainfall Features",
        color="white"
    )

    ax.tick_params(colors="white")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(
        bars,
        importance_plot_df["Mean_Absolute_SHAP"]
    ):
        ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            color="white",
            fontsize=9
        )

    ax.set_xlim(
        0,
        importance_plot_df["Mean_Absolute_SHAP"].max() * 1.18
    )

    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


    st.divider()

    # Individual SHAP explanations
    st.subheader("Individual Prediction Explanation")

    st.write(
        "Select a historical holdout observation "
        "to examine its feature contributions."
    )

    shap_columns = [
        f"SHAP_{feature}" for feature in FEATURES
    ]

    missing_shap_columns = [
        column for column in shap_columns
        if column not in shap_df.columns
    ]

    if missing_shap_columns:

        st.error(
            "Some SHAP feature columns are missing: "
            + ", ".join(missing_shap_columns)
        )

    else:

        option_labels = []

        for _, row in shap_df.iterrows():

            option_labels.append(
                f"{row['FireName']} | "
                f"{row['UniqueObservationID']} | "
                f"Actual: {int(row['Actual_Response'])} | "
                f"Predicted: {int(row['Predicted_Response'])}"
            )

        selected_position = st.selectbox(
            "Select observation",
            options=range(len(shap_df)),
            format_func=lambda i: option_labels[i]
        )

        selected_row = shap_df.iloc[selected_position]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Wildfire",
            selected_row["FireName"]
        )

        c2.metric(
            "Actual Response",
            int(selected_row["Actual_Response"])
        )

        c3.metric(
            "Predicted Response",
            int(selected_row["Predicted_Response"])
        )

        st.metric(
            "Model-estimated Probability",
            f"{float(selected_row['Predicted_Probability']) * 100:.2f}%"
        )

        contribution_df = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP_Contribution": [
                selected_row[column]
                for column in shap_columns
            ]
        })

        contribution_df["Absolute_Contribution"] = (
            contribution_df["SHAP_Contribution"].abs()
        )

        contribution_df = contribution_df.sort_values(
            "Absolute_Contribution",
            ascending=False
        )

        
    # Individual SHAP Contributions - Improved Dark Design

    plot_df = contribution_df.sort_values(
        "SHAP_Contribution"
    ).copy()

    plot_df["Display_Feature"] = (
        plot_df["Feature"]
        .map(feature_display_names)
        .fillna(plot_df["Feature"])
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    values = plot_df["SHAP_Contribution"]

    bar_colors = [
        "#F05252" if value > 0 else "#3B82F6"
        for value in values
    ]

    bars = ax.barh(
        plot_df["Display_Feature"],
        values,
        color=bar_colors,
        height=0.62
    )

    ax.axvline(
        0,
        color="white",
        linewidth=1,
        alpha=0.7
    )

    ax.set_title(
        "Individual Feature Contributions",
        color="white",
        fontsize=15,
        pad=16
    )

    ax.set_xlabel(
        "SHAP Contribution (Log-Odds)",
        color="white"
    )

    ax.set_ylabel(
        "Features",
        color="white"
    )

    ax.tick_params(colors="white")
    ax.grid(axis="x", alpha=0.18, color="white")
    ax.set_axisbelow(True)

    max_abs = max(abs(values.min()), abs(values.max()))
    offset = max_abs * 0.025 if max_abs > 0 else 0.01

    for bar, value in zip(bars, values):

        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color="white",
            fontsize=9
        )

    ax.set_xlim(-max_abs * 1.22, max_abs * 1.22)

    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.caption(
        "Red: pushes the model towards Response = 1. "
        "Blue: pushes the model towards Response = 0. "
        "Contributions are expressed in log-odds space."
    )

    with st.expander("View individual SHAP values"):

        display_contributions = contribution_df[
            ["Feature", "SHAP_Contribution"]
        ].copy()

        display_contributions["Feature"] = (
            display_contributions["Feature"]
            .map(feature_display_names)
            .fillna(display_contributions["Feature"])
        )

        st.dataframe(
            display_contributions.round(5),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 7. PAGE 4: GEOSPATIAL ANALYSIS
# =====================================================

elif page == "Geospatial Analysis":

    st.title("Geospatial Risk Analysis")

    st.write(
        "Explore historical wildfire observations and "
        "the geographical distribution of model predictions."
    )

    st.info(
        "These maps show recorded observation locations "
        "and model assessment results. They are not "
        "continuous hazard maps or live alerts."
    )

    map_files = {
        "Overall Observation Map":
            "afterburn_holdout_observation_map.html",

        "False Negative Map":
            "false_negative_map.html",

        "False Positive Map":
            "false_positive_map.html"
    }

    selected_map = st.selectbox(
        "Select Geospatial Map",
        list(map_files.keys())
    )

    map_path = OUTPUT_DIR / map_files[selected_map]

    if map_path.exists():

        map_html = map_path.read_text(
            encoding="utf-8"
        )

        components.html(
            map_html,
            height=700,
            scrolling=True
        )

    else:

        st.error(
            f"Map file not found: {map_path.name}"
        )

    st.divider()

    # State-wise performance
    st.subheader("State-wise Model Performance")

    try:

        state_df = load_output_csv(
            "state_wise_performance.csv"
        )

        st.dataframe(
            state_df.round(4),
            use_container_width=True
        )

    except Exception as error:

        st.warning(
            f"Unable to load state results: {error}"
        )

    st.divider()

    # Wildfire-wise errors
    st.subheader("Wildfire-wise Error Analysis")

    try:

        fire_df = load_output_csv(
            "wildfire_error_summary.csv"
        )

        fire_df = fire_df.sort_values(
            "Total_Errors",
            ascending=False
        )

        st.dataframe(
            fire_df.head(15).round(4),
            use_container_width=True
        )

    except Exception as error:

        st.warning(
            f"Unable to load wildfire results: {error}"
        )


# =====================================================
# 8. FOOTER
# =====================================================

st.divider()

st.caption(
    "AfterBurn Sentinel | Explainable Machine Learning "
    "Research Prototype | Historical USGS Data"
)
