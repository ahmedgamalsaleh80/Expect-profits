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

    # تنظيف أسماء الأعمدة
    df.columns = df.columns.str.strip()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())


    # ─────────────────────────────
    # AUTO COLUMN DETECTION
    # ─────────────────────────────
    def find_col(options):
        for col in options:
            if col in df.columns:
                return col
        return None


    revenue_col = find_col(["Revenue", "revenue", "Sales", "sales"])
    expenses_col = find_col(["Expenses", "expenses", "Cost", "cost"])
    marketing_col = find_col(["Marketing_Cost", "Marketing", "marketing_cost"])
    customers_col = find_col(["Num_Customers", "Customers", "customers"])
    prev_profit_col = find_col(["Previous_Profit", "Prev_Profit", "previous_profit"])
    profit_col = find_col(["Profit", "profit", "Net_Profit"])


    required_found = [
        revenue_col,
        expenses_col,
        marketing_col,
        customers_col,
        prev_profit_col,
        profit_col
    ]


    if any(col is None for col in required_found):
        st.error("❌ Dataset structure not compatible with required financial features")
        st.write("Missing columns detected:")
        st.write({
            "Revenue": revenue_col,
            "Expenses": expenses_col,
            "Marketing": marketing_col,
            "Customers": customers_col,
            "Previous Profit": prev_profit_col,
            "Profit": profit_col
        })
        st.stop()


    # ─────────────────────────────
    # FEATURES & LABELS
    # ─────────────────────────────
    X = df[[revenue_col, expenses_col, marketing_col, customers_col, prev_profit_col]].values
    y = df[profit_col].values
    y_cls = (y > 0).astype(int)


    # ─────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = (X_max - X_min) + 1e-8

    X_scaled = (X - X_min) / X_range


    # ─────────────────────────────
    # TRAIN MODELS
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
    # PREDICTIONS
    # ─────────────────────────────
    y_pred = reg.predict(X_scaled)
    cls_pred = svm.predict(X_scaled)


    # ─────────────────────────────
    # KPI SECTION
    # ─────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Profit", f"{y[y > 0].sum():,.0f}")
    col2.metric("Total Loss", f"{y[y <= 0].sum():,.0f}")
    col3.metric("Accuracy", f"{np.mean(cls_pred == y_cls):.2%}")
    col4.metric("Anomalies", len(dbscan.get_anomalies()))

    st.divider()


    # ─────────────────────────────
    # CHARTS
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
    # NEW PREDICTION
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