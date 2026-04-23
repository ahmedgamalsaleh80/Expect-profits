# ============================================================
#  Profit Prediction Dashboard — Streamlit App
#  SVM + K-Means من الصفر (بدون scikit-learn)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# ── إعداد الصفحة ──
st.set_page_config(page_title="Profit Prediction", page_icon="📊", layout="wide")
st.title("📊 Profit Prediction Dashboard")
st.markdown("**SVM + K-Means** — مبني من الصفر بدون scikit-learn")

# ============================================================
#  الدوال الأساسية (نفس المنطق بالضبط)
# ============================================================

def detect_numeric_cols(df):
    return [c for c in df.columns
            if pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df) * 0.5]

def prepare_data(df, features, target):
    sub = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
    return sub[features].values, sub[target].values, sub

def normalise(X):
    means, stds = X.mean(0), X.std(0)
    stds[stds == 0] = 1
    return (X - means) / stds

def svm_train(X, y, lr=0.01, reg=0.01, iters=300):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) < 1:
                w += lr * (yi * xi - 2*reg*w); b += lr * yi
            else:
                w -= lr * 2*reg*w
    return w, b

def svm_predict(w, b, X):
    return np.where(X.dot(w) + b >= 0, 1, -1)

def k_means(X, k, iters=100):
    centroids = X[random.sample(range(len(X)), k)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.sum((X[:, None] - centroids)**2, axis=2), axis=1)
        for ci in range(k):
            if (labels == ci).any():
                centroids[ci] = X[labels == ci].mean(0)
    return labels, centroids

def evaluate(preds, y):
    tp = ((preds==1)&(y==1)).sum(); fp = ((preds==1)&(y==-1)).sum()
    fn = ((preds==-1)&(y==1)).sum()
    acc  = (preds==y).mean()
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return acc, prec, rec, f1

# ============================================================
#  رفع الملف
# ============================================================

uploaded = st.file_uploader("📂 ارفع ملف CSV", type=["csv"])

if uploaded is None:
    st.info("👆 ارفع ملف CSV عشان يبدأ التحليل")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"✓ تم تحميل **{len(df)}** صف و **{len(df.columns)}** عمود")

# ── اكتشاف الأعمدة الرقمية ──
cols = detect_numeric_cols(df)
if len(cols) < 2:
    st.error("⚠ محتاج على الأقل عمودين رقميين!"); st.stop()

# ============================================================
#  إعدادات في الـ Sidebar
# ============================================================

st.sidebar.header("⚙ الإعدادات")

features = st.sidebar.multiselect(
    "اختار الـ Features",
    options=cols,
    default=cols[:-1]
)
target = st.sidebar.selectbox(
    "اختار الـ Target",
    options=cols,
    index=len(cols)-1
)
k        = st.sidebar.slider("عدد مجموعات K-Means", 2, 6, 3)
split    = st.sidebar.slider("نسبة التدريب %", 60, 90, 80) / 100
run_btn  = st.sidebar.button("🚀 ابدأ التحليل", use_container_width=True)

if not features or target in features:
    st.warning("اختار features مختلفة عن الـ target"); st.stop()

if not run_btn:
    st.info("اضغط **ابدأ التحليل** من الشريط الجانبي")
    st.stop()

# ============================================================
#  تشغيل النماذج
# ============================================================

with st.spinner("⚙ جاري التحليل..."):

    X, y, sub = prepare_data(df, features, target)
    X_norm    = normalise(X)
    tmean     = y.mean()
    y_svm     = np.where(y >= tmean, 1, -1)

    # تقسيم
    idx = list(range(len(X_norm))); random.shuffle(idx)
    n_tr = int(len(idx)*split)
    tr, te = idx[:n_tr], idx[n_tr:]

    # SVM
    w, b  = svm_train(X_norm[tr], y_svm[tr])
    preds = svm_predict(w, b, X_norm[te])
    acc, prec, rec, f1 = evaluate(preds, y_svm[te])

    # K-Means
    km_labels, _ = k_means(X_norm, k)
    c_counts = [(km_labels==i).sum() for i in range(k)]
    c_means  = [y[km_labels==i].mean() if (km_labels==i).any() else 0 for i in range(k)]

# ============================================================
#  عرض المقاييس
# ============================================================

st.subheader("📊 نتائج الـ SVM")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{acc*100:.1f}%")
m2.metric("Precision", f"{prec:.3f}")
m3.metric("Recall",    f"{rec:.3f}")
m4.metric("F1 Score",  f"{f1:.3f}")

st.divider()

# ============================================================
#  الرسوم البيانية
# ============================================================

fig = plt.figure(figsize=(13, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# مخطط 1: Donut
ax1 = fig.add_subplot(gs[0, 0])
n_profit = (preds==1).sum()
ax1.pie([n_profit, len(preds)-n_profit], labels=['Profit','Loss'],
        colors=['#1D9E75','#E24B4A'], autopct='%1.1f%%', wedgeprops=dict(width=0.5))
ax1.set_title(f'SVM — Profit vs Loss\nAcc:{acc*100:.1f}%  F1:{f1:.3f}')

# مخطط 2: K-Means bars
ax2 = fig.add_subplot(gs[0, 1])
colors = ['#378ADD','#EF9F27','#D4537E','#1D9E75','#7F77DD'][:k]
bars = ax2.bar([f'C{i+1}' for i in range(k)], c_counts, color=colors)
for bar, cm in zip(bars, c_means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'avg={cm:.1f}', ha='center', fontsize=8)
ax2.set_title('K-Means — Cluster Distribution')
ax2.set_ylim(0, max(c_counts)*1.2)

# مخطط 3: Feature Importance
ax3 = fig.add_subplot(gs[1, 0])
imp = np.abs(w) / (np.abs(w).max() or 1)
idx2 = np.argsort(imp)[::-1]
ax3.barh([features[i] for i in idx2], imp[idx2], color='#185FA5')
ax3.set_xlim(0, 1); ax3.invert_yaxis()
ax3.set_title('Feature Importance (SVM Weights)')

# مخطط 4: Actual vs Predicted
ax4 = fig.add_subplot(gs[1, 1])
n_plot = min(80, len(te))
target_te = y[te]
ax4.scatter(target_te[:n_plot], target_te[:n_plot], alpha=0.5, color='#378ADD', s=25)
lims = [target_te.min(), target_te.max()]
ax4.plot(lims, lims, '--', color='#E24B4A', lw=1.5)
ax4.set_xlabel('Actual'); ax4.set_ylabel('Predicted')
ax4.set_title('Actual vs Predicted')

st.pyplot(fig)

# ============================================================
#  جدول التنبؤات
# ============================================================

st.subheader("🗂 عينة من التنبؤات (أول 20 صف)")
result_df = sub.iloc[te[:20]].copy().reset_index(drop=True)
result_df['SVM_Pred'] = ['✅ Profit' if p==1 else '❌ Loss' for p in preds[:20]]
result_df['Cluster']  = [f'C{km_labels[i]+1}' for i in te[:20]]
st.dataframe(result_df, use_container_width=True)

st.success("✅ اكتمل التحليل!")
