import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import joblib

from src.preprocessing import load_data
from src.simulation import simulate
from src.optimization import optimize

st.title("🍬 Factory Optimization System")

df = load_data(os.path.join(BASE_DIR, 'data/orders.csv'))
model = joblib.load(os.path.join(BASE_DIR, 'models/best_model.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'models/encoders.pkl'))

# UI
product = st.selectbox("Select Product", df['Product Name'].unique())
priority = st.slider("Speed vs Profit", 0.0, 1.0, 0.5)

# Simulation
sim_df = simulate(df, product, model, encoders)

st.subheader("📊 Simulation Results")
st.dataframe(sim_df)

# Optimization
opt_df = optimize(sim_df, weight_time=priority, weight_profit=(1 - priority))

st.subheader("🏆 Top Recommendations")
st.dataframe(opt_df.head(3))

# Charts
st.subheader("📈 Lead Time")
st.bar_chart(opt_df.set_index('factory')['lead_time'])

st.subheader("💰 Profit")
st.bar_chart(opt_df.set_index('factory')['profit'])