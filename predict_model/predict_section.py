import streamlit as st
import numpy as np
import joblib
import pandas as pd
from predict_model.streamlit_demo_app import render

@st.cache_data
def load_week_to_predict():
    return pd.read_csv("predict_model/WeekToPredict.csv")

@st.cache_resource
def load_models():
    return {
        "arb_clf_rf": joblib.load("predict_model/arb_clf_rf.joblib"),
        "arb_clf_xgb_gate": joblib.load("predict_model/arb_clf_xgb_noprob.joblib"),
        "arb_lin_rf": joblib.load("predict_model/arb_lin_rf.joblib"),
        "can_clf_rf": joblib.load("predict_model/can_clf_rf.joblib"),
        "can_lin_rf": joblib.load("predict_model/can_lin_rf.joblib"),
        "flu_clf_rf": joblib.load("predict_model/flu_clf_rf.joblib"),
        "flu_lin_rf": joblib.load("predict_model/flu_reg_rf.joblib"),
        "hep_clf_xgb": joblib.load("predict_model/hep_clf_xgb.joblib"),
        "hep_lin_rf": joblib.load("predict_model/hep_lin_rf.joblib"),
        "tub_clf_xgb": joblib.load("predict_model/tub_clf_xgb.joblib"),
        "tub_lin_rf": joblib.load("predict_model/tub_lin_rf.joblib"),
        "mea_clf_xgb": joblib.load("predict_model/mea_clf_xgb.joblib"),
        "mea_clf_xgb_gate": joblib.load("predict_model/mea_clf_xgb_noprob.joblib"),
        "mea_lin_rf": joblib.load("predict_model/mea_lin_rf.joblib"),
    }

def render_nowcast_section():

    df = load_week_to_predict()
    models = load_models()

    render()