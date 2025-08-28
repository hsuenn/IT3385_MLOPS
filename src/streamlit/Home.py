"""
streamlit multi-page apps
"""
import streamlit as st
from omegaconf import OmegaConf

# load config file
cfg = OmegaConf.load("configs/streamlit.yaml") # relative from cwd

st.set_page_config(page_title=cfg.home.interface.page_title, page_icon=cfg.home.interface.page_icon, layout="wide")