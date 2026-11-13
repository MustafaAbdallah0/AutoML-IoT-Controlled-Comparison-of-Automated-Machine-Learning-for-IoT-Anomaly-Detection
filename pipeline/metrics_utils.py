"""metrics_utils.py - identical metric computation for every framework and baseline."""
import numpy as np
from sklearn import metrics as M


def compute_metrics(y_true, y_pred, labels):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    p, r, f, s = M.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": float(M.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(M.balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(M.f1_score(y_true, y_pred, labels=labels, average="weighted",
                                        zero_division=0)),
        "mcc": float(M.matthews_corrcoef(y_true, y_pred)),
    }
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                     "support": int(s[i])} for i, c in enumerate(labels)}
    cm = M.confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out, per_class, cm


def permutation_importance(predict_fn, X, y, labels, n=3000, repeats=3, seed=0):
    """Model-agnostic permutation importance = mean drop in macro-F1 when a feature is shuffled."""
    rng = np.random.RandomState(seed)
    if len(X) > n:
        idx = rng.choice(len(X), n, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    base = compute_metrics(y, predict_fn(X), labels)[0]["macro_f1"]
    imp = {}
    for col in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - compute_metrics(y, predict_fn(Xp), labels)[0]["macro_f1"])
        imp[col] = float(np.mean(drops))
    return {"baseline_macro_f1": float(base), "importance": imp}
