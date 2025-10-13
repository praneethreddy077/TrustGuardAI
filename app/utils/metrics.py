import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "f1": float(f1), "auc": float(auc), "cm": cm}
