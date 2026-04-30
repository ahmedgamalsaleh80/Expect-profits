import numpy as np


def select_features(df):
    """
    Extract features and target variables from dataframe.
    """
    X = df[["Revenue", "Expenses", "Marketing_Cost", "Num_Customers", "Previous_Profit"]].values
    y_reg = df["Profit"].values
    y_cls = df["Profit_Label"].values
    return X, y_reg, y_cls


def train_test_split_manual(X, y, test_size=0.2, random_seed=42):
    """
    Manual train-test split (no sklearn).
    """
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X_train, X_test):
    """
    Min-Max normalization.
    Returns normalized data + parameters.
    """
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)

    X_range = (X_max - X_min)
    X_range[X_range == 0] = 1e-8  # avoid division by zero

    X_train_scaled = (X_train - X_min) / X_range
    X_test_scaled = (X_test - X_min) / X_range

    return X_train_scaled, X_test_scaled, X_min, X_range


def preprocess(df):
    """
    Full preprocessing pipeline.
    """
    X, y_reg, y_cls = select_features(df)

    X_train, X_test, y_train_reg, y_test_reg = train_test_split_manual(X, y_reg)
    _, _, y_train_cls, y_test_cls = train_test_split_manual(X, y_cls)

    X_train_scaled, X_test_scaled, X_min, X_range = normalize(X_train, X_test)

    return {
        "X_train_reg": X_train_scaled,
        "X_test_reg": X_test_scaled,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "X_train_cls": X_train_scaled,
        "X_test_cls": X_test_scaled,
        "y_train_cls": y_train_cls,
        "y_test_cls": y_test_cls,
        "X_full": (X - X_min) / X_range,
        "y_reg": y_reg,
        "y_cls": y_cls,
    }