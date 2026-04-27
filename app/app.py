import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import joblib
import pandas as pd

from src.preprocessing import load_data
from src.simulation import simulate
from src.optimization import optimize

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Factory Optimization System",
    layout="wide"
)

# ------------------ LOAD DATA ------------------
df = load_data(os.path.join(BASE_DIR, 'data/orders.csv'))
model = joblib.load(os.path.join(BASE_DIR, 'models/best_model.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'models/encoders.pkl'))

# ------------------ SIDEBAR ------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Dashboard",
    "Simulation",
    "Recommendations"
])

# ------------------ DASHBOARD ------------------
if page == "Dashboard":
    st.title("Factory Optimization Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Products", df['Product Name'].nunique())
    col2.metric("Orders", len(df))
    col3.metric("Regions", df['Region'].nunique())

    st.markdown("---")

    st.subheader("📈 Sales by Region")
    st.bar_chart(df.groupby('Region')['Sales'].sum())

# ------------------ SIMULATION ------------------
elif page == "Simulation":
    st.title("⚙️ Factory Simulation")

    tab1, tab2, tab3 = st.tabs([
        "🔹 Input",
        "📊 Results",
        "📈 Charts"
    ])

    # ---------- INPUT ----------
    with tab1:
        st.subheader("Select Inputs")

        product = st.selectbox("Select Product", df['Product Name'].unique())
        priority = st.slider("Speed vs Profit", 0.0, 1.0, 0.5)

        run = st.button("Run Simulation")

    if 'sim_df' not in st.session_state:
        st.session_state.sim_df = None
        st.session_state.opt_df = None

    if run:
        sim_df = simulate(df, product, model, encoders)
        opt_df = optimize(sim_df, weight_time=priority, weight_profit=(1 - priority))

        st.session_state.sim_df = sim_df
        st.session_state.opt_df = opt_df

    # ---------- RESULTS ----------
    with tab2:
        st.subheader("Simulation Results")

        if st.session_state.sim_df is not None:
            st.dataframe(st.session_state.sim_df, use_container_width=True)
        else:
            st.info("Run simulation to see results")

    # ---------- CHARTS ----------
    with tab3:
        st.subheader("Performance Charts")

        if st.session_state.opt_df is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Lead Time")
                st.bar_chart(
                    st.session_state.opt_df.set_index('factory')['lead_time']
                )

            with col2:
                st.subheader("Profit")
                st.bar_chart(
                    st.session_state.opt_df.set_index('factory')['profit']
                )
        else:
            st.info("Run simulation to see charts")

# ------------------ RECOMMENDATIONS ------------------
elif page == "Recommendations":
    st.title("🏆 Factory Recommendations")

    if 'opt_df' in st.session_state and st.session_state.opt_df is not None:

        best = st.session_state.opt_df.iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Factory", best['factory'])
        col2.metric("Lead Time", round(best['lead_time'], 2))
        col3.metric("Profit", round(best['profit'], 2))

        st.markdown("---")

        st.subheader("Top 3 Recommendations")
        st.dataframe(st.session_state.opt_df.head(3), use_container_width=True)

    else:
        st.warning("⚠️ Run simulation first")