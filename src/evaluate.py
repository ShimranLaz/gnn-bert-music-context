import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score

def evaluate_predictions(y_true, y_probs, threshold=0.20):
    y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_probs_np = y_probs.detach().cpu().numpy() if isinstance(y_probs, torch.Tensor) else y_probs
    y_pred_np = (y_probs_np >= threshold).astype(int)

    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true_np, y_pred_np, average='micro', zero_division=0)

    auc_pr_scores = []
    for k in range(y_true_np.shape[1]):
        if len(np.unique(y_true_np[:, k])) > 1:
            auc_pr_scores.append(average_precision_score(y_true_np[:, k], y_probs_np[:, k]))
    auc_pr = np.mean(auc_pr_scores) if len(auc_pr_scores) > 0 else 0.0

    return macro_f1, micro_f1, auc_pr

def compute_recall_at_k(sim_mat, k_values=[1, 5, 10]):
    n = sim_mat.shape[0]
    recalls = {}
    for k in k_values:
        actual_k = min(k, n)
        top_k_preds = np.argsort(-sim_mat, axis=1)[:, :actual_k]
        hits = sum([1 for i in range(n) if i in top_k_preds[i]])
        recalls[f"R@{k}"] = hits / n
    return recalls

if __name__ == "__main__":
    with open("results/metrics.json") as f:
        print(f.read())
