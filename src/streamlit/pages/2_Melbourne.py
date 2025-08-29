"""
frontend page for Melbourne residential pricing prediction
for ease of editing, please open up configs/streamlit.yaml side by side
"""
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from src.models.melbourne.model import get_model, make_prediction

# load config file
cfg = OmegaConf.load("configs/streamlit.yaml") # relative from cwd
cfg = cfg.melbourne # narrow down scope to reduce repetitiveness

# load model (wrap in a function to use cache decorator, to prevent re-runs)
@st.cache_resource(show_spinner=False)
def _get_model_cached(model_path=cfg.model_path):
    """
    model_path: str, file path of saved pipeline relative to cwd (root)

    file path should be stripped of `.pkl` extension
    without extension as pycaret load_model adds in the `.pkl` automatically 
    allow user to specify different model

    returns:
        saved_pipeline: sklearn.pipeline.Pipeline
    """
    return get_model(cfg.model_path)

# page config
st.set_page_config(
    page_title=cfg.interface.page_title,
    page_icon=cfg.interface.page_icon,
    layout="wide"
)


# helper function
def log_predictions_to_disk(pred_df, model_fp, log_file_fp):
    """
    pred_df: pd.DataFrame, new predicted data to log
    model_fp: str, relative path to model file, wo `.pkl` directory
    log_file_fp: str, relative path to .csv file containing logs, same value as cfg.models.pred_log_path

    will log input columns and predicted columns too
    will append new rows into log_file_fp

    returns
        None
    """
    # append model path key to pred_df
    pred_df["model_path"] = model_fp

    # read current logs if any
    if os.path.exists(log_file_fp):
        # exists
        current_log_df = pd.read_csv(log_file_fp)

        # concatenate (append to bottom)
        new_log_df = pd.concat((
            current_log_df,
            pred_df
        ))
    else:
        new_log_df = pred_df

    # save to file
    new_log_df.to_csv(log_file_fp, index=False)

def build_form_from_input(cols, values=None):
    """
    cols: [c1: st.delta_generator.DeltaGenerator, c2, c3], return value of st.columns(3) function
    values: { [key: str]: float|int }|None, where key in cfg.model.features, user supplied file, will populate inputs with default values from values

    builds the form to modify a single row input

    returns
        data: pd.DataFrame, contains original values, overridden by user input
        errs: { [key: str]: str }, where key in cfg.model.features, lsit all the input errors (if values arg was supplied)
    """
    data = {} # build data while iterating through features
    errs = {}
    for idx, feat in enumerate(cfg.model.features):
        col = cols[idx %len(cols)] # len(cols) == 3
        feat_params = cfg.model.features_range[feat.lower()] # parameters for the features (e.g. default, range, step and help)

        # only use user-supplied value if type matches
        default_value = \
            (values and feat in values and type(values[feat]) == type(feat_params["default"])) \
            or feat_params["default"] # use values first (e.g. user-supplied file) over pre-defined default values, TODO: sanity check input values

        if (type(feat_params["default"]) in [int, float]):
            # numerical input
            mini, maxi = feat_params["range"] # input range

            # apply sanity check to ensure within range
            if (mini <= default_value <= maxi):
                # valid input
                val = col.number_input(feat, mini, maxi, default_value, feat_params["step"], help=feat_params["help"])
            else:
                # use default value
                val = col.number_input(feat, mini, maxi, feat_params["default"], feat_params["step"], help=feat_params["help"])
                errs[feat] = "File supplied value ({}) falls outside of range [{}, {}] (inclusive).".format(default_value, mini, maxi)
        elif (type(feat_params["default"]) == str):
            if ("choices" in feat_params):
                # choices
                # see if user supplied value exists
                index = 0
                if default_value in feat_params["choices"]:
                    index = feat_params["choices"].index(default_value) # guaranteed to exist since condition above checks for it, hence no need to catch for ValueError
                else:
                    # add to error for verbosity
                    errs[feat] = "File supplied value ({}) falls outside of choices [{}] (inclusive).".format(default_value, ", ".join(feat_params["choices"]))
                col.selectbox(feat, list(feat_params["choices"]), index=index, help=feat_params["help"])
            else:
                # free-text input
                val = col.text_input(feat, value=default_value.strip()) # strip user-supplied input (from file, also another round of stripping when using this field input)
                val = val.strip() # strip input from user (this time round its for keyed in input, whereas above was for file input)

        # enter field
        data[feat] = val

    return pd.DataFrame([data]), errs

def coerce_batch_schema(df):
    """
    df: pd.DataFrame, user input dataframe

    1. keeps only features required (drop unexpected columns)
    2. converts date column to YearSold and MonthSold (drops date column)
    3. fill in missing rows and columns with default values
    4. re-order columns to match training exactly

    returns
        formatted_df: pd.DataFrame, df ready for prediction
    """
    df = df.copy()

    # parse date if present
    if ("Date" in df.columns):
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # add in YearSold and MonthSold values
            if ("YearSold" not in df.columns):
                df["YearSold"] = df["Date"].dt.year.fillna(cfg.model.features_range.yearsold.default)
            if ("MonthSold" not in df.columns):
                df["MonthSold"] = df["Date"].dt.month.fillna(cfg.model.features_range.monthsold.default)
        except Exception:
            st.warning("Could not parse 'Date' column. It will be dropped.")
            df = df.drop(columns=["Date"])

    # drop unexpected columns
    unexpected_cols = [c for c in df.columns if c not in cfg.model.features]
    if unexpected_cols:
        st.warning(f"Dropping unexpected columns: {', '.join(unexpected_cols)}")
        df = df.drop(columns=unexpected_cols)

    # create new columns if expected columns are missing
    # fill in missing rows and columns with default values
    for feat in cfg.model.features:
        # reference default value from streamlit.yaml
        default_val = cfg.model.features_range[feat.lower()].default

        # missing column, create new column
        if (feat not in df.columns):
            df[feat] = default_val

        # missing rows, fill in with default
        df[feat] = df[feat].fillna(default_val)

    # re-order columns to match training exactly
    return df[cfg.model.features]


# -----------------------
# Sidebar — configuration
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Set model source and provide a sample CSV (schema) to auto-build the form.")

    default_model_path = cfg.model_path # without extension as pycaret load_model adds in the `.pkl` automatically
    model_path = st.text_input("MODEL_PATH", value="{}.pkl".format(default_model_path), placeholder="e.g., melbourne.pkl") # ask user to input `.pkl` extension for completenesss, will strip out later

    st.divider()
    sample_csv = st.file_uploader("Upload SAMPLE CSV (optional, small file)", type=["csv"])

    st.divider()
    log_predictions = st.toggle("Log predictions to CSV", value=True)
    if log_predictions:
        st.caption("Logging to {}".format(cfg.model.pred_log_path))

    st.divider()
    st.markdown("**About**")
    st.caption("IT3385 AY2025 S1 • Task 3 — Real-Time Prediction App • Streamlit")

# -----------------------
# Main layout
# -----------------------
st.title("{} {}".format(cfg.interface.page_icon, cfg.interface.page_title))
st.write("Enter feature values on the left, get instant predictions on the right.")

# reference cached model
try:
    fp_wo_ext = ".".join(model_path.strip().split(".")[:-1]) # strip user supplied .pkl extension
    model = _get_model_cached(fp_wo_ext)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model not loaded yet: {e}")

    # important else will throw error if below codes are ran
    st.stop() # since model not defined, no point loading rest of the UI

# tabs, 2 modes, single vs batch
tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])
with tab_single:
    st.subheader("Input features")

    user_df = None
    if sample_csv is not None:
        # user uploaded file
        user_df = pd.read_csv(sample_csv)

        # take first row, convert to single dict
        user_df = user_df.iloc[0].to_dict()

    # build user input to allow user to modify submitted fields
    # otherwise, build user inputs from supplied defaults (in streamlit.yaml) for immediate prediction
    cols = st.columns(3)
    user_df, errs = build_form_from_input(cols, user_df) # if passed in user_df is None, function will supply default values defined in config file (streamlit.yaml)

    # user uploaded file as input AND file contains illegal values
    if (sample_csv and errs):
        for err in errs.keys():
            st.error(errs[err]) # show error message on why failed to use file inputs

    st.divider()
    st.write("### Prediction")
    pred_df = make_prediction(model, user_df)

    # KPI card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Result")

        st.metric("Prediction", "${:.2f}".format(pred_df["Predicted"].iloc[0]))

    # log prediction if user toggled button
    if log_predictions:
        # concat old logs and save to disk
        log_path = cfg.model.pred_log_path
        log_predictions_to_disk(pred_df, cfg.model_path, log_path)

        # toast notification
        st.toast("Saved to {}".format(log_path), icon="✅")

with tab_batch:
    st.subheader("Upload a CSV for batch predictions")
    st.caption("Use the **same columns** as your training set. The app will append a prediction column.")

    batch_file = st.file_uploader("Choose CSV", type=["csv"], key="batch_upl")
    if batch_file is not None:
        try:
            # read uploaded file
            batch_df = pd.read_csv(batch_file)

            # coerce columns to required inputs
            batch_df = coerce_batch_schema(batch_df)

            st.write("Preview:", batch_df.head())
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            batch_df = None
    else:
        batch_df = None

    if st.button("Run Batch Prediction", disabled=(batch_df is None)):
        pred_df = make_prediction(model, batch_df)

        # show predictions
        st.write("Prediction:", pred_df.head())

        # offer download
        buf = io.BytesIO()
        pred_df.to_csv(buf, index=False)
        buf.seek(0)
        st.success("Batch prediction complete.")
        st.download_button("Download results CSV", data=buf, file_name="predictions.csv", mime="text/csv")

        # log prediction if user toggled button
        if log_predictions:
            # concat old logs and save to disk
            log_path = cfg.model.pred_log_path
            log_predictions_to_disk(pred_df, cfg.model_path, log_path)

            # toast notification
            st.toast("Saved to {}".format(log_path), icon="✅")

st.divider()
with st.container():
    st.markdown("##### Debug / Model Info")
    with st.expander("Show model repr()"):
        try:
            st.code(repr(model)[:5000])
        except Exception:
            st.write("(unavailable)")
