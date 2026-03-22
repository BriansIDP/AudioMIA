import json
import sys, os
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def get_tpr_at_fpr(y_true, y_scores, target_fpr=0.05):
    # Get the FPR and TPR at various thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the index where FPR is closest to or just under the target
    idx = np.where(fpr <= target_fpr)[0][-1]
    return tpr[idx]

infile = sys.argv[1]
outfile = infile.replace(".json", "_results.json")

with open(infile) as fin:
    data = json.load(fin)

labels = []
all_metrics = {}

for datapiece in data:
    labels.append(datapiece["label"])
    for metric, value in datapiece["pred"].items():
        if metric not in all_metrics:
            all_metrics[metric] = []
        all_metrics[metric].append(value)

results = {}
categorised_results = {}
for metric, values in all_metrics.items():
    if metric == "pred":
        continue
    score = roc_auc_score(labels, values)
    tpr_at_low_fpr = get_tpr_at_fpr(labels, values)
    print("AUROC {}:".format(metric), score)
    print("TPR @5% FPR {}:".format(metric), tpr_at_low_fpr)
    results[metric] = {"auroc": score, "tpr_low_fpr": tpr_at_low_fpr}

with open(outfile, "w") as fout:
    json.dump(results, fout, indent=4)