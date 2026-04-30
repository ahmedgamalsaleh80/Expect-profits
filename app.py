import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
from svm import LinearSVM
from kmeans import KMeans
from dbscan import DBSCAN


# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(page_title="Profit Dashboard", layout="centered")

st.title("📊 Profit Analytics Dashboard (From Scratch ML)")
st.write("Upload your company dataset to analyze profit patterns")

# ─────────────────────────────
# UPLOAD DATA
# ─────────────────────────────
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ─────────────────────────────
    # VALIDATION
    # ─────────────────────────────
    required_cols = [
        "Revenue",
        "Expenses",
        "Marketing_Cost",
        "Num_Customers",
        "Previous_Profit",
        "Profit"
    ]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV missing required columns")
        st.stop()

    # ─────────────────────────────
    # FEATURES & LABELS
    # ─────────────────────────────
    X = df[[
        "Revenue",
        "Expenses",
        "Marketing_Cost",
        "Num_Customers",
        "Previous_Profit"
    ]].values

    y = df["Profit"].values
    y_cls = (y > 0).astype(int)

    # ─────────────────────────────
    # NORMALIZATION (FROM SCRATCH)
    # ─────────────────────────────
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = (X_max - X_min) + 1e-8

    X_scaled = (X - X_min) / X_range

    # ─────────────────────────────
    # TRAIN MODELS (FROM SCRATCH ONLY)
    # ─────────────────────────────
    st.subheader("🧠 Training Models...")

    reg = LinearRegression(learning_rate=0.01, n_iterations=600)
    reg.fit(X_scaled, y)

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iterations=400)
    svm.fit(X_scaled, y_cls)

    kmeans = KMeans(k=3, max_iterations=80)
    kmeans.fit(X_scaled)

    dbscan = DBSCAN(epsilon=0.5, min_samples=4)
    dbscan.fit(X_scaled[:, :2])

    st.success("✅ Models trained successfully")

    # ─────────────────────────────
    # PREDICTIONS ON DATASET
    # ─────────────────────────────
    y_pred = reg.predict(X_scaled)
    cls_pred = svm.predict(X_scaled)

    # ─────────────────────────────
    # KPI SECTION (CLEAN & SMALL)
    # ─────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Profit", f"{y[y > 0].sum():,.0f}")
    col2.metric("Total Loss", f"{y[y <= 0].sum():,.0f}")
    col3.metric("Accuracy", f"{np.mean(cls_pred == y_cls):.2%}")
    col4.metric("Anomalies", len(dbscan.get_anomalies()))

    st.divider()

    # ─────────────────────────────
    # SMALL CHARTS
    # ─────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.caption("📈 Actual vs Predicted Profit")

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(y[:25], label="Actual", linewidth=1)
        ax.plot(y_pred[:25], label="Predicted", linewidth=1)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        st.pyplot(fig)

    with c2:
        st.caption("📊 Profit / Loss Distribution")

        fig2, ax2 = plt.subplots(figsize=(3, 2))
        ax2.bar(["Profit", "Loss"], [sum(y > 0), sum(y <= 0)])
        ax2.tick_params(labelsize=6)
        st.pyplot(fig2)

    # ─────────────────────────────
    # NEW INPUT PREDICTION
    # ─────────────────────────────
    st.divider()
    st.subheader("🔮 Predict New Month")

    revenue = st.number_input("Revenue", value=100000)
    expenses = st.number_input("Expenses", value=50000)
    marketing = st.number_input("Marketing Cost", value=5000)
    customers = st.number_input("Number of Customers", value=200)
    prev_profit = st.number_input("Previous Profit", value=20000)

    if st.button("🚀 Predict"):

        x = np.array([[revenue, expenses, marketing, customers, prev_profit]])
        x_scaled = (x - X_min) / X_range

        profit = reg.predict(x_scaled)[0]
        cls = svm.predict(x_scaled)[0]
        cluster = kmeans.predict(x_scaled)[0]

        st.subheader("📌 Result")

        st.metric("Predicted Profit", f"{profit:,.2f}")

        if cls == 1:
            st.success("Profit Month ✅")
        else:
            st.error("Loss Month ❌")

        st.info(f"Cluster ID: {cluster}")

else:
    st.info("⬆️ Please upload a CSV file to start analysis")