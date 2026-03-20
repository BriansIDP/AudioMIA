import json
import sys, os
from sklearn.metrics import roc_auc_score


infile = sys.argv[1]

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

for metric, values in all_metrics.items():
    score = roc_auc_score(labels, values)
    print("AUROC {}:".format(metric), score)