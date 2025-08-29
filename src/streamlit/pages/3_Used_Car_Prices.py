"""
frontend page for used car prices model
for ease of editing, please open up configs/streamlit.yaml side by side
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import seaborn as sns

from src.models.used_car_prices.model import get_model, make_prediction


# load config file
cfg = OmegaConf.load("configs/streamlit.yaml") # relative from cwd
cfg = cfg.used_car_prices # narrow down scope to reduce repetitiveness

# XGBOOST BACKWARD-COMPAT PATCH
# Lets older GPU-trained XGB pickles load & run on CPU-only envs
if not hasattr(xgb.XGBModel, "gpu_id"):
    xgb.XGBModel.gpu_id = None
if not hasattr(xgb.XGBModel, "predictor"):
    xgb.XGBModel.predictor = None
if not hasattr(xgb.XGBModel, "device"):
    xgb.XGBModel.device = None

_OLD_SET_PARAMS = xgb.XGBModel.set_params
def _set_params_safely(self, **params):
    # Drop unknown / GPU-only params
    for k in ["gpu_id", "gpu_id_ref", "predictor", "device", "n_gpus", "use_label_encoder"]:
        params.pop(k, None)
    return _OLD_SET_PARAMS(self, **params)
xgb.XGBModel.set_params = _set_params_safely

# HELPER FUNCTION CACHE MODEL
@st.cache_resource(show_spinner=False)
def _load_pipeline(model_path=cfg.model_path):
    return get_model(cfg.model_path) # if file path does not exist, will throw an error internally

# HELPERS
def cast_frame_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to expected dtypes + keep only model features in order."""
    df = df.copy()
    for c in CAT_COLS:
        if c in df.columns: df[c] = df[c].astype("category")
    for c in INT_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype("int64")
    for c in FLOAT_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df[[c for c in FEATURES if c in df.columns]]

def to_float(name, s, min_val=0.0):
    try:
        v = float(str(s).strip())
        if v < min_val:
            return None, f"{name} must be ≥ {min_val}"
        return v, None
    except ValueError:
        return None, f"{name} must be a number"

# FEATURES / COL DEFINES
FEATURES = cfg.model.features
CAT_COLS = cfg.model.cat_cols
INT_COLS = cfg.model.int_cols
FLOAT_COLS = cfg.model.float_cols

# TRAINING METRICS (CV mean) - Taken from Task 2
with open(cfg.train_metrics_fp, "r") as f:
    TRAIN_METRICS = json.load(f)


# PAGE APPEARANCE
st.set_page_config(
    page_title=cfg.interface.page_title,
    page_icon=cfg.interface.page_icon,
    layout="wide"
)


with open(cfg.custom_styling_fp, "r") as f:
    CUSTOM_STYLING_ATTR = f.read()
st.markdown(CUSTOM_STYLING_ATTR, unsafe_allow_html=True)

# HEADER
st.markdown("<h1 style='margin-bottom:0'>{} {}</h1>".format(cfg.interface.page_icon, cfg.interface.page_title), unsafe_allow_html=True)
st.caption("Powered by your PyCaret regression pipeline.")

# SIDEBAR
model = None
with st.sidebar:
    st.subheader("Model")
    model_name = st.text_input("Saved pipeline name", value="{}.pkl".format(cfg.model_path)) # ask user to input `.pkl` extension for completenesss, will strip out later

    # LOAD MODEL
    try:
        model = _load_pipeline(model_name)
        st.success("Model loaded ✔")
    except AssertionError as err:
        # filepath not found
        st.error("Could not load model: {}".format(err))

        # important, since model not ready, no point rendering rest of the work
        st.stop()

    st.write("---")
    mode = st.radio("Prediction mode", ["Single", "Batch (CSV)"], horizontal=False)
    st.info("Training features:\n[{}]".format(", ".join(FEATURES)))

# SINGLE MODE
if mode == "Single":

    # Headline card
    st.markdown(
        """
        <div class="card card--light">
          <h3 style="margin:0">Enter the car details <span class="subtle">(then hit <span class="accent">Predict</span>)</span></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Inputs in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    # reference descriptors for each features
    features_range = cfg.model.features_range
    with col1:
        location = st.selectbox("Location", list(features_range.location.choices), index=cfg.model.features_range.location.choices.index(features_range.location.default))
        kms = st.number_input("Kilometers Driven", min_value=features_range.kilometers_driven.range[0], value=features_range.kilometers_driven.default, step=features_range.kilometers_driven.step)
        fuel = st.selectbox("Fuel Type", list(features_range.fuel_type.choices), index=features_range.fuel_type.choices.index(features_range.fuel_type.default))
        transmission = st.selectbox("Transmission", list(features_range.transmission.choices), index=features_range.transmission.choices.index(features_range.transmission.default))
        owner = st.selectbox("Owner Type", list(features_range.owner_type.choices), index=features_range.owner_type.choices.index(features_range.owner_type.default))

    with col2:
        seats = st.selectbox("Seats", list(features_range.seats.choices), index=features_range.seats.choices.index(features_range.seats.default))
        mileage_str = st.number_input("Mileage (km/kg)", min_value=features_range.mileage_km_kg.range[0], value=features_range.mileage_km_kg.default, help="Enter a decimal number")
        power_str = st.number_input("Power (bhp)", min_value=features_range.power_bhp.range[0], value=features_range.power_bhp.default, help="Enter a decimal number")
        car_age = st.number_input("Car age", min_value=features_range.car_age.range[0], value=features_range.car_age.default, step=features_range.car_age.step)
    st.markdown('</div>', unsafe_allow_html=True)

    # Validate numerics
    mileage, err_mileage = to_float("Mileage (km/kg)", mileage_str, features_range.mileage_km_kg.range[0])
    power, err_power = to_float("Power (bhp)", power_str, features_range.power_bhp.range[0])
    if err_mileage: st.warning(err_mileage)
    if err_power:   st.warning(err_power)

    can_predict = (model is not None) and (err_mileage is None) and (err_power is None)

    if st.button("Predict", type="primary", disabled=not can_predict):
        X = pd.DataFrame([{
            "Location": location,
            "Kilometers_Driven": int(kms),
            "Fuel_Type": fuel,
            "Transmission": transmission,
            "Owner_Type": owner,
            "Seats": int(seats),
            "Mileage_km_kg": mileage,
            "Power_bhp": power,
            "Car_Age": car_age,
        }])
        X = cast_frame_types(X)
        print("X", X)

        try:
            out_df = make_prediction(model, X) # cols=[..., "Predicted"]
            yhat = float(out_df["Predicted"].iloc[0])

            # Result card
            st.markdown('<div class="card result-card">', unsafe_allow_html=True)
            st.markdown("#### Predicted Price (INR Lakhs)")
            st.markdown(f"<div class='big-price'>₹ {yhat:,.2f}</div>", unsafe_allow_html=True)
            st.markdown('<div class="subtle">Based on the features you provided</div>', unsafe_allow_html=True)
            with st.expander("View input row sent to the model"):
                st.dataframe(X, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Model Quality vs. This Prediction (3 separate charts)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("## Model Quality vs. This Prediction")

            # Pull CV metrics; fall back if missing
            mse_cv = TRAIN_METRICS.get("MSE")
            rmse_cv = TRAIN_METRICS.get("RMSE")
            r2_cv = TRAIN_METRICS.get("R2")

            if rmse_cv is None and mse_cv is not None:
                rmse_cv = float(mse_cv) ** 0.5

            # Helper: convert rupees to lakhs
            def to_lakhs(x):
                return None if x is None else float(x) / 1e5

            # 1) RMSE (Lakhs)
            if rmse_cv is not None:
                fig_rmse, ax_rmse = plt.subplots(figsize=(6, 3.8))
                train_val = to_lakhs(rmse_cv)
                single_est = to_lakhs(rmse_cv)  # uncertainty scale for one prediction
                sns.barplot(
                    x=["Training (CV mean)", "Single input"],
                    y=[train_val, single_est],
                    palette=["#0ea5e9", "#84cc16"],
                    ax=ax_rmse,
                )
                ax_rmse.set_ylabel("RMSE (Scaled)")
                ax_rmse.set_title("RMSE comparison")
                st.pyplot(fig_rmse)
            else:
                st.info("RMSE not found in TRAIN_METRICS (will only show MSE/R² if present).")

            # 2) MSE (Lakhs²) – divide rupee MSE by (1e5)² = 1e10 (For scaling)
            if mse_cv is not None:
                fig_mse, ax_mse = plt.subplots(figsize=(6, 3.8))
                train_val = float(mse_cv) / 1e10
                single_est = (float(rmse_cv) ** 2) / 1e10 if rmse_cv is not None else None
                y_vals = [train_val, single_est]
                x_labels = ["Training (CV mean)", "Single input"]
                sns.barplot(x=x_labels, y=y_vals, palette=["#0ea5e9", "#84cc16"], ax=ax_mse)
                ax_mse.set_ylabel("MSE (Scaled)")
                ax_mse.set_title("MSE comparison")
                st.pyplot(fig_mse)

            # Optional: display a 1-σ band around your prediction using RMSE
            if rmse_cv is not None:
                lower = yhat - rmse_cv
                upper = yhat + rmse_cv
                st.caption(
                    f"Approximate 1-σ uncertainty band based on CV RMSE: **₹ {lower:,.0f} to ₹ {upper:,.0f}**."
                    f"NOTE: Laks^2 was used to rescale the target variable"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction or chart generation failed: {e}")

# BATCH MODE
else:
    # Upload instructions card
    st.markdown('<div class="card card--light">', unsafe_allow_html=True)
    st.markdown("### Upload a CSV with the exact columns:")
    st.code(", ".join(FEATURES), language="text")
    up = st.file_uploader("Choose CSV file", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if up is not None:
        try:
            df_in = pd.read_csv(up)

            # tolerant renames
            soft_map = {
                "Mileage (km/kg)": "Mileage_km_kg",
                "Power (bhp)"    : "Power_bhp",
            }
            for old, new in soft_map.items():
                if old in df_in.columns and new not in df_in.columns:
                    df_in = df_in.rename(columns={old: new})

            # validate
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            # cast & order
            Xb = cast_frame_types(df_in)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Preview (first 5 rows)")
            st.dataframe(Xb.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Predict Batch", type="primary"):
                if model is None:
                    st.error("Load a model first from the sidebar.")
                else:
                    try:
                        out_df = make_prediction(model, Xb) # cols=[..., "Predicted"]

                        results = Xb.copy()
                        results["Predicted_Price_INR_Lakhs"] = out_df["Predicted"].astype(float)

                        # Results + download card
                        st.markdown('<div class="card result-card">', unsafe_allow_html=True)
                        st.markdown("#### Results (first 20 rows)")
                        st.dataframe(results.head(20), use_container_width=True)

                        csv_bytes = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv_bytes,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Charts card
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("## Batch Insights")

                        # 1) Histogram of predicted prices
                        st.markdown("### Distribution of Predicted Prices")
                        fig_h, ax_h = plt.subplots()
                        sns.histplot(results["Predicted_Price_INR_Lakhs"], kde=True, ax=ax_h)
                        ax_h.set_xlabel("Predicted Price (INR Lakhs)")
                        ax_h.set_ylabel("Count")
                        st.pyplot(fig_h)

                        # 2) Average predicted price by Location (Top 10)
                        if "Location" in results.columns:
                            st.markdown("### Top Locations by Average Predicted Price (Top 10)")
                            loc_mean = (
                                results.groupby("Location")["Predicted_Price_INR_Lakhs"]
                                .mean()
                                .sort_values(ascending=False)
                                .head(10)
                                .reset_index()
                            )
                            fig_loc, ax_loc = plt.subplots()
                            sns.barplot(data=loc_mean, x="Predicted_Price_INR_Lakhs", y="Location", ax=ax_loc)
                            ax_loc.set_xlabel("Avg Predicted Price (INR Lakhs)")
                            ax_loc.set_ylabel("Location")
                            st.pyplot(fig_loc)

                        # 3) Power vs Predicted Price (colored by Fuel_Type if present)
                        if {"Power_bhp", "Predicted_Price_INR_Lakhs"}.issubset(results.columns):
                            st.markdown("### Power vs Predicted Price")
                            fig_s, ax_s = plt.subplots()
                            if "Fuel_Type" in results.columns:
                                sns.scatterplot(
                                    data=results,
                                    x="Power_bhp",
                                    y="Predicted_Price_INR_Lakhs",
                                    hue="Fuel_Type",
                                    alpha=0.6,
                                    ax=ax_s
                                )
                                ax_s.legend(title="Fuel Type", bbox_to_anchor=(1.05, 1), loc="upper left")
                            else:
                                sns.scatterplot(
                                    data=results,
                                    x="Power_bhp",
                                    y="Predicted_Price_INR_Lakhs",
                                    alpha=0.6,
                                    ax=ax_s
                                )
                            ax_s.set_xlabel("Power (bhp)")
                            ax_s.set_ylabel("Predicted Price (INR Lakhs)")
                            st.pyplot(fig_s)

                        # 4) Predicted price by Fuel_Type (boxplot), if available
                        if {"Fuel_Type", "Predicted_Price_INR_Lakhs"}.issubset(results.columns):
                            st.markdown("### Predicted Price by Fuel Type")
                            fig_bx, ax_bx = plt.subplots()
                            sns.boxplot(
                                data=results,
                                x="Fuel_Type",
                                y="Predicted_Price_INR_Lakhs",
                                ax=ax_bx
                            )
                            ax_bx.set_xlabel("Fuel Type")
                            ax_bx.set_ylabel("Predicted Price (INR Lakhs)")
                            st.pyplot(fig_bx)

                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

        except Exception as e:
            st.error(f"Could not read the file: {e}")