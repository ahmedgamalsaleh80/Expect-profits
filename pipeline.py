"""
pipeline.py
-----------
Stable ML Pipeline (From Scratch Models)
"""

import numpy as np
from data_generator import generate_data
from preprocessing import preprocess
from linear_regression import LinearRegression
from svm import LinearSVM
from kmeans import KMeans
from dbscan import DBSCAN


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_pipeline():

    # =========================
    # STEP 1: DATA
    # =========================
    print_header("STEP 1: Generating Dataset")

    df = generate_data(n_samples=500, random_seed=42)

    print(f"Rows: {df.shape[0]}")
    print(f"Profit months: {int(df['Profit_Label'].sum())}")
    print(f"Loss months: {int(len(df) - df['Profit_Label'].sum())}")

    # =========================
    # STEP 2: PREPROCESSING
    # =========================
    print_header("STEP 2: Preprocessing")

    data = preprocess(df)

    X_train_reg = data['X_train_reg']
    y_train_reg = data['y_train_reg']
    X_test_reg  = data['X_test_reg']
    y_test_reg  = data['y_test_reg']

    X_train_cls = data['X_train_cls']
    y_train_cls = data['y_train_cls']
    X_test_cls  = data['X_test_cls']
    y_test_cls  = data['y_test_cls']

    X_full = data['X_full']
    y_reg  = data['y_reg']

    print("Preprocessing completed ✔")

    # =========================
    # STEP 3: LINEAR REGRESSION
    # =========================
    print_header("STEP 3: Linear Regression")

    reg_model = LinearRegression(
        learning_rate=0.01,
        n_iterations=1500
    )

    reg_model.fit(X_train_reg, y_train_reg)

    reg_metrics = reg_model.evaluate(X_test_reg, y_test_reg)

    print("Results:")
    print(f"R2   = {reg_metrics.get('R2')}")
    print(f"MAE  = {reg_metrics.get('MAE')}")
    print(f"RMSE = {reg_metrics.get('RMSE')}")

    preds = reg_model.predict(X_test_reg[:5])

    print("\nSample Predictions:")
    for p, a in zip(preds, y_test_reg[:5]):
        print(f"Pred: {p:.0f} | Actual: {a:.0f}")

    # =========================
    # STEP 4: SVM
    # =========================
    print_header("STEP 4: SVM Classification")

    svm_model = LinearSVM(
        learning_rate=0.0005,
        lambda_param=0.01,
        n_iterations=1200
    )

    svm_model.fit(X_train_cls, y_train_cls)

    svm_metrics = svm_model.evaluate(X_test_cls, y_test_cls)

    print("Results:")
    print(f"Accuracy  = {svm_metrics['Accuracy']:.2%}")
    print(f"Precision = {svm_metrics['Precision']:.2%}")
    print(f"Recall    = {svm_metrics['Recall']:.2%}")
    print(f"F1 Score  = {svm_metrics['F1_Score']:.2%}")

    # =========================
    # STEP 5: KMEANS
    # =========================
    print_header("STEP 5: K-Means")

    kmeans_model = KMeans(k=3, max_iterations=100)
    kmeans_model.fit(X_full)

    inertia = kmeans_model.inertia(X_full)

    print(f"Inertia = {inertia}")

    kmeans_model.cluster_summary(X_full, y_reg)

    # =========================
    # STEP 6: DBSCAN
    # =========================
    print_header("STEP 6: DBSCAN")

    X_2d = X_full[:, :2]

    dbscan_model = DBSCAN(epsilon=0.5, min_samples=4)
    dbscan_model.fit(X_2d)

    dbscan_model.summary(df)

    # =========================
    # FINAL SUMMARY
    # =========================
    print_header("FINAL SUMMARY")

    print(f"""
Linear Regression R2: {reg_metrics.get('R2')}
SVM Accuracy: {svm_metrics['Accuracy']:.2%}
KMeans Clusters: 3
DBSCAN Anomalies: {len(dbscan_model.get_anomalies())}
""")


if __name__ == "__main__":
    run_pipeline()