# ============================================================
#  Profit Prediction — SVM + K-Means (from scratch)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random

# ── 1. تحميل البيانات ──
def load_csv(path):
    df = pd.read_csv(path)
    print(f"✓ {len(df)} صف | {len(df.columns)} عمود")
    return df

# ── 2. اكتشاف الأعمدة الرقمية (أكثر من 50% أرقام) ──
def detect_numeric_cols(df):
    cols = [c for c in df.columns
            if pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df) * 0.5]
    print(f"✓ أعمدة رقمية: {cols}")
    return cols

# ── 3. تنظيف البيانات وإرجاع X, y ──
def prepare_data(df, features, target):
    sub = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
    print(f"✓ صفوف صالحة: {len(sub)}")
    return sub[features].values, sub[target].values, sub

# ── 4. تطبيع Z-Score: (x - mean) / std ──
def normalise(X):
    means, stds = X.mean(0), X.std(0)
    stds[stds == 0] = 1
    return (X - means) / stds

# ── 5. تدريب SVM بـ SGD ──
# score = w·x + b ؛ لو y*score < 1 → حدّث بالكامل، غير كده → regularisation فقط
def svm_train(X, y, lr=0.01, reg=0.01, iters=300):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) < 1:
                w += lr * (yi * xi - 2*reg*w)
                b += lr * yi
            else:
                w -= lr * 2*reg*w
    return w, b

# ── 6. تنبؤ SVM: score >= 0 → +1 (Profit)، غير كده → -1 (Loss) ──
def svm_predict(w, b, X):
    return np.where(X.dot(w) + b >= 0, 1, -1)

# ── 7. K-Means: قسّم البيانات لـ k مجموعات ──
def k_means(X, k, iters=100):
    centroids = X[random.sample(range(len(X)), k)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.sum((X[:, None] - centroids)**2, axis=2), axis=1)
        for ci in range(k):
            if (labels == ci).any():
                centroids[ci] = X[labels == ci].mean(0)
    return labels, centroids

# ── 8. تقييم الـ SVM ──
def evaluate(preds, y):
    tp = ((preds==1) & (y==1)).sum()
    fp = ((preds==1) & (y==-1)).sum()
    fn = ((preds==-1) & (y==1)).sum()
    acc  = (preds == y).mean()
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return acc, prec, rec, f1

# ── 9. رسم الـ Dashboard (4 مخططات) ──
def plot_results(preds, y_test, target_test, km_labels, k,
                 c_counts, c_means, tmean, features, w, acc, f1):
    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # مخطط 1: Donut — Profit vs Loss
    ax1 = fig.add_subplot(gs[0, 0])
    n_profit = (preds == 1).sum()
    ax1.pie([n_profit, len(preds)-n_profit], labels=['Profit','Loss'],
            colors=['#1D9E75','#E24B4A'], autopct='%1.1f%%', wedgeprops=dict(width=0.5))
    ax1.set_title(f'SVM — Profit vs Loss\nAcc: {acc*100:.1f}%  F1: {f1:.3f}')

    # مخطط 2: K-Means cluster sizes
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
    idx = np.argsort(imp)[::-1]
    ax3.barh([features[i] for i in idx], imp[idx], color='#185FA5')
    ax3.set_xlim(0, 1); ax3.invert_yaxis()
    ax3.set_title('Feature Importance (SVM Weights)')

    # مخطط 4: Actual vs Predicted
    ax4 = fig.add_subplot(gs[1, 1])
    n = min(80, len(target_test))
    ax4.scatter(target_test[:n], target_test[:n], alpha=0.5, color='#378ADD', s=25)
    lims = [target_test.min(), target_test.max()]
    ax4.plot(lims, lims, '--', color='#E24B4A', lw=1.5)
    ax4.set_xlabel('Actual'); ax4.set_ylabel('Predicted')
    ax4.set_title('Actual vs Predicted')

    plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
    print("✓ تم حفظ dashboard.png")
    plt.show()

# ── MAIN ──
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    try:
        df = load_csv(path)
    except FileNotFoundError:
        print(f"⚠ مش لاقي الملف: {path}"); return

    cols = detect_numeric_cols(df)
    if len(cols) < 2:
        print("⚠ محتاج عمودين رقميين على الأقل!"); return

    features, target = cols[:-1], cols[-1]
    print(f"\n📌 Features: {features}\n📌 Target:   {target}")

    X, y, sub = prepare_data(df, features, target)
    X_norm     = normalise(X)

    # Labels: فوق المتوسط = Profit (+1)، تحت = Loss (-1)
    tmean = y.mean()
    y_svm = np.where(y >= tmean, 1, -1)

    # تقسيم 80/20
    idx = list(range(len(X_norm))); random.shuffle(idx)
    n_tr = int(len(idx)*0.8)
    tr, te = idx[:n_tr], idx[n_tr:]

    print(f"\n✓ تدريب: {len(tr)} | اختبار: {len(te)}")

    print("\n⚙  تدريب SVM...")
    w, b = svm_train(X_norm[tr], y_svm[tr])
    preds = svm_predict(w, b, X_norm[te])
    acc, prec, rec, f1 = evaluate(preds, y_svm[te])

    print(f"\n📊 SVM:\n  Accuracy={acc*100:.1f}%  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    print(f"\n⚙  K-Means (k=3)...")
    km_labels, _ = k_means(X_norm, k=3)
    c_counts = [(km_labels==i).sum() for i in range(3)]
    c_means  = [y[km_labels==i].mean() if (km_labels==i).any() else 0 for i in range(3)]

    print("\n📊 K-Means:")
    for i in range(3):
        tag = "High" if c_means[i] >= tmean else "Low"
        print(f"  C{i+1}: {c_counts[i]} صف | avg={c_means[i]:.2f} | {tag}")

    plot_results(preds, y_svm[te], y[te], km_labels, 3,
                 c_counts, c_means, tmean, features, w, acc, f1)
    print("\n✅ اكتمل!")

if __name__ == "__main__":
    main()
