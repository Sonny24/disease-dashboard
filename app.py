import streamlit as st
import pandas as pd
import plotly.express as px
import us
from compartmental_models import comp_model

st.set_page_config(page_title="CDC Weekly Disease Dashboard", layout="wide")
st.title("CDC Weekly Disease Transmission Dashboard")

# TODO: ADJUST INTRO FOR MODELS 
st.markdown("""

This dashboard provides an interactive view of weekly case counts for **12 priority infectious disease groups** 
across the United States.  
You can:
- View national trends by disease group
- View disease activity week-by-week across the U.S
- Compare state-level patterns side-by-side
- Run prediction models on diseases
- Review peak weekly case counts for each disease group
            
**Please note that all models used have a degree of uncertainty. Use multiple sources to make any vital decisions.**
""")

df = pd.read_csv("model_data/NNDSS_Weekly_Data_20251110.csv")

df = df[
    [
        "Reporting Area",
        "Current MMWR Year",
        "MMWR WEEK",
        "Label",
        "Current week",
        "LOCATION1"
    ]
].dropna(subset=["MMWR WEEK", "Label", "Current week"])

df["MMWR WEEK"] = pd.to_numeric(df["MMWR WEEK"], errors="coerce")
df["Current week"] = pd.to_numeric(df["Current week"], errors="coerce")

DISEASE_KEYWORDS = {
    "Viral Hemorrhagic Fevers": ["hemorrhagic fever"],
    "Zika Virus": ["zika"],
    "Tuberculosis": ["tuberculosis"],
    "Measles": ["measles"],
    "Hepatitis A": ["hepatitis a"],
    "Hepatitis B": ["hepatitis b"],
    "Hepatitis C (acute)": ["hepatitis c, acute"],
    "Novel Influenza A": ["novel influenza a"],
    "Anthrax (all types)": ["anthrax"],
    "West Nile Virus": ["west nile"],
    "Candida auris": ["candida auris"],
    "Botulism": ["botulism"]
}

def map_disease(label):
    label_lower = str(label).lower()
    for disease, keywords in DISEASE_KEYWORDS.items():
        if any(k in label_lower for k in keywords):
            return disease
    return None

df["Disease Group"] = df["Label"].apply(map_disease)
df = df.dropna(subset=["Disease Group"])


total_rows = df[df["Reporting Area"].str.upper() == "TOTAL"].copy()


total_rows = total_rows[
    ["Current MMWR Year", "MMWR WEEK", "Disease Group", "Current week"]
]


national_data = (
    total_rows
    .groupby(["Current MMWR Year", "MMWR WEEK", "Disease Group"], as_index=False)
    ["Current week"]
    .sum()
)


year = st.selectbox(
    "Select MMWR Year",
    sorted(national_data["Current MMWR Year"].unique()),
    index=0
)

filtered = national_data[national_data["Current MMWR Year"] == year]


st.subheader(f"Weekly Disease Cases — United States ({year})")

fig = px.line(
    filtered,
    x="MMWR WEEK",
    y="Current week",
    color="Disease Group",
    markers=True,
    title=f"Weekly Disease Cases — {year}"
)

fig.update_layout(
    xaxis_title="MMWR Week",
    yaxis_title="Current Week Cases",
    template="plotly_white",
    hovermode="x unified",
    legend_title="Disease"
)

st.plotly_chart(fig, width='stretch')


states = [state.name for state in us.states.STATES]

# US MAP
st.subheader("U.S. Map — State-Level Cases")

disease_list = sorted(df["Disease Group"].unique())
selected_disease = st.selectbox("Select Disease", disease_list)

week_list = sorted(df["MMWR WEEK"].unique())
selected_week = st.selectbox("Select Week", week_list)

map_data = df[
    (df["Disease Group"] == selected_disease) &
    (df["MMWR WEEK"] == selected_week) &
    (df["Reporting Area"] != "TOTAL")
].copy()

state_summary = (
    map_data.groupby("LOCATION1")["Current week"]
    .sum()
    .reset_index()
)

def to_abbrev(x):
    try:
        return us.states.lookup(x).abbr
    except:
        return None

state_summary["abbr"] = state_summary["LOCATION1"].apply(lambda x: to_abbrev(x))
state_summary = state_summary.dropna(subset=["abbr"])

fig = px.scatter_geo(
    state_summary,
    locations="abbr",
    locationmode="USA-states",
    size="Current week",
    color="Current week",
    hover_name="abbr",
    hover_data={"Current week": True},
    scope="usa",
    title=f"{selected_disease} — Week {selected_week}",
)

fig.update_layout(
    geo=dict(
        scope="usa",
        projection_type="albers usa",
        showland=True
    ),
    coloraxis_colorbar_title="Cases"
)

st.plotly_chart(fig, width='stretch')

st.subheader("State-Level Comparison")
col1, col2 = st.columns(2)

with col1:
    state1 = st.selectbox("Select State (Left Chart)", states, index=0)
with col2:
    state2 = st.selectbox("Select State (Right Chart)", states, index=1)


def make_state_fig(state):
    sub = df[(df["LOCATION1"].str.upper() == state.upper()) & (df["Current MMWR Year"] == year)]
    fig = px.line(
        sub,
        x="MMWR WEEK",
        y="Current week",
        color="Disease Group",
        markers=True,
        title=f"Weekly Disease Cases — {state} ({year})"
    )
    fig.update_layout(
        xaxis_title="MMWR Week",
        yaxis_title="Current Week Cases",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Disease"
    )
    return fig

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(make_state_fig(state1), width='stretch')
with col4:
    st.plotly_chart(make_state_fig(state2), width='stretch')

selected_state = st.selectbox("Select State", states)

tb_fig = comp_model.model_treatment_paper(
    selected_state,
    weeks=30,
    y_log=True,
    plot=True
)

# NOTE: doesn't work with any other state besides maryland because lack of data
zika_fig = comp_model.run_zika_pipeline(
    selected_state,
    weeks_back=100,
    forecast_weeks=104,
    obs_model="incidence",
    reporting_rate=0.8
)

measle_fig = comp_model.run_measles_pipeline(
    selected_state,
    weeks_back=80,
    forecast_weeks=104,
    obs_model="incidence",
    reporting_rate=0.8
)

hepB_fig = comp_model.run_hepB_pipeline(
    state=selected_state,
    weeks_back=10, 
    weeks_forward=10
)

tb_tab, zika_tab, measles_tab, hepb_tab = st.tabs(["Tuberculosis", "Zika", "Measles", "Hepatitis B"])

with tb_tab:
    st.header("Tuberculosis Projection")
    if tb_fig is not None:
        st.pyplot(tb_fig)
    else:
        st.warning("No data available for this selection")
with zika_tab:
    if zika_fig is not None:
        st.pyplot(zika_fig)
    else:
        st.warning("No data available for this selection")
with measles_tab:
    if measle_fig is not None:
        st.pyplot(measle_fig)
    else:
        st.warning("No data available for this selection")
with hepb_tab:
    if hepB_fig is not None:
        st.pyplot(hepB_fig)
    else:
        st.warning("No data available for this selection")

st.write("### Summary Table — Peak Weekly Values")
summary = (
    filtered.groupby("Disease Group")["Current week"]
    .max()
    .reset_index()
    .rename(columns={"Current week": "Max Weekly Cases"})
    .sort_values("Max Weekly Cases", ascending=False)
)
st.dataframe(summary, hide_index=True)