"""
frontend page for wheat classification model
for ease of editing, please open up configs/streamlit.yaml side by side
"""
import os
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from src.models.wheat.model import get_model, make_prediction


# load config file
cfg = OmegaConf.load("configs/streamlit.yaml") # relative from cwd
cfg = cfg.wheat # narrow down scope to reduce repetitiveness

# load model
model = get_model(cfg.model_path)

# page configuration
st.set_page_config(page_title=cfg.interface.page_title, page_icon=cfg.interface.page_icon, layout="wide")

# set app header
st.markdown("## 🌾 Wheat Kernel Classifier")
st.caption("Classify Wheat Types Based on Kernel Attributes Using a PyCaret Machine Learning Model")
st.divider()

# define model schema
FEATURES = cfg.model.features
FEATURES_DESC = cfg.model.features_desc # in same order
FEATURES_DEFAULT = cfg.model.features_default
ID2MAP = cfg.id2label

# offer two modes
# 1. single prediction
# 2. batch prediction
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio(
    "Select Prediction Mode:",
    ["Single Prediction", "Batch Prediction (CSV)"],
    index=0
)
st.caption("IT3385 AY2025 S1 • Task 3 — Real-Time Prediction App • Streamlit")


if mode == "Single Prediction":
    # single prediction mode
    st.subheader("🔍 Provide the Features of the Wheat Kernel:")

    # create form
    with st.form("single_form", clear_on_submit=False):
        cols = st.columns(3)
        vals = {}
        errs = []

        for feat_idx, feat_name in enumerate(FEATURES):
            with cols[feat_idx %3]:
                inp = st.text_input(
                    feat_name,
                    value="{:.2f}".format(FEATURES_DEFAULT[feat_idx]),
                    help=FEATURES_DESC[feat_idx]
                )
                try:
                    # sanity check
                    vals[feat_name] = float(inp.strip())
                    if vals[feat_name] < 0:
                        # beyond negative
                        raise ValueError
                except ValueError:
                    # either input is not numeric or value is negative
                    errs.append("{} must be a valid non-negative number".format(feat_name)) # non-negative to include 0

        # predict button (inside form)
        submitted = st.form_submit_button("Predict")
        if submitted:
            # button is clicked
            if (len(errs) >= 1):
                # has errors
                st.error(" | ".join(errs))
            else:
                inp_df = pd.DataFrame([vals])
                predictions = make_prediction(model, inp_df)
                if not predictions.empty:
                    # obtain predicted label and corresponding confidence score
                    predicted_wheat_type = int(predictions["Predicted"].iloc[0])
                    confidence = predictions["Confidence"].iloc[0] *100

                    # map id back to name
                    wheat_type_name = ID2MAP[predicted_wheat_type -1] # predicted_wheat_type is one-based indexing, convert to zero-based

                    # output success
                    st.success("✅ Prediction Completed!")
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted Wheat Type:", "🌾 Type {} ({})".format(predicted_wheat_type, wheat_type_name))
                    c2.metric("Confidence (%):", "{:.2f}%".format(confidence))
else:
    # batch prediction
    st.subheader("📂 Batch Predictions")
    st.write("Upload a CSV File Containing the Following Columns (Order is Not Important):")
    st.code(", ".join(FEATURES))

    # allow user to download dummy data
    with open(cfg.demo_data_fp, "r", encoding="utf-8") as f:
        dummy_data_contents = f.read()
    st.caption("If you don’t have a CSV file containing the required columns, you can use this sample template to get started:")
    st.download_button(
        "⬇️ Download Sample Template CSV",
        data=dummy_data_contents,
        # data=_demo_batch().to_csv(index=False).encode("utf-8"),
        file_name="Wheat_Seeds_Template.csv",
        mime="text/csv",
    )

    st.write("")

    # upload csv
    uploaded_file = st.file_uploader("Upload Own CSV File", type=["csv"])
    if uploaded_file:
        inp_df = pd.read_csv(uploaded_file)
        missing = [f for f in FEATURES if f not in inp_df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            st.info(f"✅ {uploaded_file.name} Uploaded Successfully. Preview of First 5 Rows:")
            st.dataframe(inp_df.head(5))

            with st.spinner("Generating Predictions....."):
                predictions = make_prediction(model, inp_df)

            if not predictions.empty:
                predictions["Wheat_Name"] = predictions["Predicted"].map(lambda predicted_idx: ID2MAP[predicted_idx -1]) # predicted_idx is one-based indexing, convert to zero-based

                st.success(f"Batch Predictions Generated! Here are the Results of the Predictions for {uploaded_file.name} :")
                st.dataframe(predictions, use_container_width=True)

                # display batch Summary
                st.subheader("📊 Batch Prediction Summary")
                summary = (
                    predictions.groupby(["Predicted", "Wheat_Name"])
                    .agg(
                        Count=("Predicted", "size"),
                        Average_Confidence=("Confidence", "mean"),
                    )
                    .reset_index()
                )
                summary["Average_Confidence"] = (summary["Average_Confidence"] * 100).round(2)

                # bar chart Sorted by Count (descending)
                import altair as alt
                count_chart = (
                    alt.Chart(summary.sort_values("Count", ascending=False))
                    .mark_bar()
                    .encode(
                        y=alt.Y("Count:Q", title="Number of Samples"),
                        x=alt.X(
                            "Wheat_Name:N",
                            sort="-y",
                            title="Wheat Type",
                            axis=alt.Axis(labelAngle=45)  # Rotate labels 45 degrees
                        ),
                        color=alt.Color(
                            "Wheat_Name:N",
                            scale=alt.Scale(range=["#F5BABB", "#568F87", "#FED16A"]),
                            legend=None
                        ),
                        tooltip=[
                            alt.Tooltip("Wheat_Name:N", title="Predicted Wheat Type:"),
                            alt.Tooltip("Count:Q", title="Predicted Total Samples:"),
                            alt.Tooltip("Average_Confidence:Q", title="Average Confidence (%):", format=".2f")
                        ]
                    )
                )

                st.altair_chart(count_chart, use_container_width=True)

                # download Full Predictions
                st.download_button(
                    "⬇️ Export Prediction Results as CSV",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="Wheat_Seeds_Predictions.csv",
                    mime="text/csv",
                )

# footer
st.divider()
st.caption("⚡ Built with PyCaret + Streamlit | Wheat Kernel Classification")