"""Evaluation helpers for multiclass intrusion classification."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def macro_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute accuracy and macro precision/recall/F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }
