import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Disease Nowcast", layout="centered")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("predict_model/WeekToPredict.csv")

# ----------------------------
# Load models
# ----------------------------
arb_clf_rf = joblib.load("predict_model/arb_clf_rf.joblib")
arb_clf_xgb_gate = joblib.load("predict_model/arb_clf_xgb_noprob.joblib")   # gate (has predict_proba)
arb_lin_rf = joblib.load("predict_model/arb_lin_rf.joblib")

can_clf_rf = joblib.load("predict_model/can_clf_rf.joblib")
can_lin_rf = joblib.load("predict_model/can_lin_rf.joblib")

flu_clf_rf = joblib.load("predict_model/flu_clf_rf.joblib")
flu_lin_rf = joblib.load("predict_model/flu_reg_rf.joblib")

hep_clf_xgb = joblib.load("predict_model/hep_clf_xgb.joblib")
hep_lin_rf = joblib.load("predict_model/hep_lin_rf.joblib")

tub_clf_xgb = joblib.load("predict_model/tub_clf_xgb.joblib")
tub_lin_rf = joblib.load("predict_model/tub_lin_rf.joblib")

mea_clf_xgb = joblib.load("predict_model/mea_clf_xgb.joblib")
mea_clf_xgb_gate = joblib.load("predict_model/mea_clf_xgb_noprob.joblib")   # gate (has predict_proba)
mea_lin_rf = joblib.load("predict_model/mea_lin_rf.joblib")

# ----------------------------
# Feature columns (MUST match training)
# ----------------------------
TRAIN_FEATURES = [
    "MMWR WEEK",
    "lag_1","lag_2","lag_3","lag_4","lag_8","lag_12","lag_16",
    "roll_mean_4","roll_mean_8","roll_std_4","roll_std_8",
    "growth_1","growth_4","pct_change_1","pct_change_4",
    "month","quarter","day_of_year",
    "season_winter","season_spring","season_summer","season_fall",
    "cases_prev",
    "neighbor_cases_sum","neighbor_cases_mean","neighbor_cases_max","neighbor_states_reporting",
    "log_population","log_density",
]
feature_cols = TRAIN_FEATURES

# ----------------------------
# App domain lists
# ----------------------------
DISEASES = [
    "Influenza",
    "Candida auris",
    "Measles",
    "Tuberculosis",
    "Hepatitis",
    "Arboviral diseases",
]

STATES = [
    "Alabama","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "Florida","Georgia","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana",
    "Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
    "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
    "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee",
    "Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming",
]

MODEL_REGISTRY = {
    "Measles": {
        "clf": mea_clf_xgb,           # probability model (display)
        "reg": mea_lin_rf,
        "gate_clf": mea_clf_xgb_gate, # used for final 0/1
        "gate_thr": 0.02,
        "prob_thr": 0.50,             # only used if no gate (not used here, but kept for consistency)
    },
    "Arboviral diseases": {
        "clf": arb_clf_rf,
        "reg": arb_lin_rf,
        "gate_clf": arb_clf_xgb_gate,
        "gate_thr": 0.05,
        "prob_thr": 0.50,
    },
    "Candida auris": {"clf": can_clf_rf, "reg": can_lin_rf, "prob_thr": 0.50},
    "Influenza":     {"clf": flu_clf_rf, "reg": flu_lin_rf, "prob_thr": 0.50},
    "Hepatitis":     {"clf": hep_clf_xgb, "reg": hep_lin_rf, "prob_thr": 0.50},
    "Tuberculosis":  {"clf": tub_clf_xgb, "reg": tub_lin_rf, "prob_thr": 0.50},
}

# ----------------------------
# Helpers
# ----------------------------
def get_nextweek_row(df_in: pd.DataFrame, disease: str, state: str):
    sub = df_in[(df_in["Disease_group"] == disease) & (df_in["Reporting Area norm"] == state)]
    if sub.empty:
        return None
    if len(sub) > 1:
        # should not happen, but safe
        sub = sub.head(1)
    return sub.copy()

def make_X(row_df: pd.DataFrame, cols):
    X = row_df.reindex(columns=cols).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

def proba_1(model, X):
    """Return P(y=1). Assumes model has predict_proba."""
    return float(model.predict_proba(X)[:, 1][0])

# ----------------------------
# UI inputs
# ----------------------------
def render():
    col1, col2 = st.columns(2)
    with col1:
        disease = st.selectbox("Disease group", DISEASES, index=DISEASES.index("Influenza"))
    with col2:
        state = st.selectbox("State", STATES, index=STATES.index("Virginia"))

    st.divider()

    cfg = MODEL_REGISTRY[disease]
    row_df = get_nextweek_row(df, disease, state)
    if row_df is None:
        st.error("No row found in WeekToPredict.csv for that disease + state.")
        st.stop()

    X_row = make_X(row_df, feature_cols)

    # ----------------------------
    # Classification (NO sliders)
    # ----------------------------
    # Always compute probability model (for display)
    p_case = proba_1(cfg["clf"], X_row)

    # Final 0/1 decision:
    # - if gate exists: use gate proba + gate_thr
    # - else: use prob model proba + prob_thr
    if "gate_clf" in cfg:
        gate_thr = float(cfg["gate_thr"])
        gate_p = proba_1(cfg["gate_clf"], X_row)
        has_case = int(gate_p >= gate_thr)
    else:
        gate_p = np.nan
        prob_thr = float(cfg.get("prob_thr", 0.50))
        has_case = int(p_case >= prob_thr)

    # ----------------------------
    # Regression
    # ----------------------------
    reg = cfg["reg"]
    pred_cases = float(reg.predict(X_row)[0])
    pred_cases = max(0.0, pred_cases)

    gate_reg = st.toggle("Gate regressor by classifier decision", value=True)
    pred_cases_display = pred_cases if (not gate_reg or has_case == 1) else 0.0

    # ----------------------------
    # Outputs
    # ----------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted cases? (0/1)", f"{has_case}")
    k2.metric("Chance of cases (prob model)", f"{p_case*100:.1f}%")
    k3.metric("Predicted # cases", f"{pred_cases_display:.1f}")

    # Always show decision text + probability (no debug tables)
    st.write(f"**Prob model:** p={p_case:.4f}")

    if "gate_clf" in cfg:
        st.write(f"**Gate model:** p={gate_p:.4f} | thr={gate_thr:.2f} | decision={has_case}")
    else:
        st.write(f"**Decision rule:** p >= {float(cfg.get('prob_thr', 0.50)):.2f} → {has_case}")
