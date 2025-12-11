import streamlit as st
import pandas as pd
import plotly.express as px

# --- Streamlit page setup ---
st.set_page_config(page_title="CDC Weekly Disease Dashboard", layout="wide")
st.title("CDC Weekly Disease Transmission Dashboard")


st.markdown("""

This dashboard provides an interactive view of weekly case counts for **12 priority infectious disease groups** 
across the United States.  
You can:
- Upload the weekly CDC NNDSS dataset (CSV)
- View **national trends** by disease group
- Compare **state-level patterns** side-by-side
- Review **peak weekly case counts** for each disease group

Upload a data file below to begin.
""")

# --- File upload ---
uploaded_file = st.file_uploader("Upload data", type=["csv"])


if uploaded_file:

    df = pd.read_csv(uploaded_file)
    
    # Keep relevant columns
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

    # Convert numeric fields
    df["MMWR WEEK"] = pd.to_numeric(df["MMWR WEEK"], errors="coerce")
    df["Current week"] = pd.to_numeric(df["Current week"], errors="coerce")

    # --- Disease filters (keywords) ---
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

    # --- Map diseases ---
    def map_disease(label):
        label_lower = str(label).lower()
        for disease, keywords in DISEASE_KEYWORDS.items():
            if any(k in label_lower for k in keywords):
                return disease
        return None

    df["Disease Group"] = df["Label"].apply(map_disease)
    df = df.dropna(subset=["Disease Group"])


    # Keep only TOTAL rows
    total_rows = df[df["Reporting Area"].str.upper() == "TOTAL"].copy()

    # Keep only the needed columns
    total_rows = total_rows[
        ["Current MMWR Year", "MMWR WEEK", "Disease Group", "Current week"]
    ]

    # Group by Year + Week + Disease Group
    national_data = (
        total_rows
        .groupby(["Current MMWR Year", "MMWR WEEK", "Disease Group"], as_index=False)
        ["Current week"]
        .sum()
    )

    # --- Year selection ---
    year = st.selectbox(
        "Select MMWR Year",
        sorted(national_data["Current MMWR Year"].unique()),
        index=0
    )

    filtered = national_data[national_data["Current MMWR Year"] == year]

    # --- Plot National Current-Week Cases ---
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

    st.plotly_chart(fig, use_container_width=True)

    # --- STATE-LEVEL FILTERS ---
    states = sorted(df["LOCATION1"].dropna().astype(str).str.upper().unique())

    st.subheader("State-Level Comparison")
    col1, col2 = st.columns(2)

    with col1:
        state1 = st.selectbox("Select State (Left Chart)", states, index=0)
    with col2:
        state2 = st.selectbox("Select State (Right Chart)", states, index=1)

    # --- STATE GRAPHS ---
    def make_state_fig(state):
        sub = df[(df["LOCATION1"].str.upper() == state) & (df["Current MMWR Year"] == year)]
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
        st.plotly_chart(make_state_fig(state1), use_container_width=True)
    with col4:
        st.plotly_chart(make_state_fig(state2), use_container_width=True)

    # --- Summary Table ---
    st.write("### Summary Table — Peak Weekly Values")
    summary = (
        filtered.groupby("Disease Group")["Current week"]
        .max()
        .reset_index()
        .rename(columns={"Current week": "Max Weekly Cases"})
        .sort_values("Max Weekly Cases", ascending=False)
    )
    st.dataframe(summary, hide_index=True)
else:
    st.info("Upload data")
