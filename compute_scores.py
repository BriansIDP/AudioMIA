import json
import sys, os
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
from bert_score import score


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

infile = sys.argv[1]
useref = False
per_audio = True
use_max = True
refdata = None
use_ratio = False
rouge_topk = 0

outfile = infile.replace(".json", "_precompute.json")

def compare_with_orig(metric, orig_metric):
    if use_ratio:
        return metric / orig_metric
    else:
        return metric - orig_metric


def compute_rouge(texts, reftext=None):
    rouge_scores = {"rouge_1": 0, "rouge_2": 0, "rouge_l": 0}
    rouge_scorelist = {"rouge_1": [], "rouge_2": [], "rouge_l": []}
    total_score_1 = []
    total_score_2 = []
    total_score_l = []
    total = 0
    for i in range(len(texts)):
        if reftext is not None:
            scores = scorer.score(texts[i].lower(), reftext.lower())
            total_score_1.append(scores["rouge1"].fmeasure)
            total_score_2.append(scores["rouge2"].fmeasure)
            total_score_l.append(scores["rougeL"].fmeasure)
        else:
            for j in range(i+1, len(texts)):
                scores = scorer.score(texts[i].lower(), texts[j].lower())
                total_score_1.append(scores["rouge1"].fmeasure)
                total_score_2.append(scores["rouge2"].fmeasure)
                total_score_l.append(scores["rougeL"].fmeasure)
    if rouge_topk > 0:
        rouge_scores["rouge_1"] = 1 - np.sort(total_score_1)[-rouge_topk:][0] / rouge_topk
        rouge_scores["rouge_2"] = 1 - np.sort(total_score_2)[-rouge_topk:][0] / rouge_topk
        rouge_scores["rouge_l"] = 1 - np.sort(total_score_l)[-rouge_topk:][0] / rouge_topk
    else:
        rouge_scores["rouge_1"] = 1 - np.mean(total_score_1)
        rouge_scores["rouge_2"] = 1 - np.mean(total_score_2)
        rouge_scores["rouge_l"] = 1 - np.mean(total_score_l)
    rouge_scorelist["rouge_1"] = total_score_1
    rouge_scorelist["rouge_2"] = total_score_2
    rouge_scorelist["rouge_l"] = total_score_l
    return rouge_scores, rouge_scorelist


def get_bert_score(texts, reftext=None):
    refs = []
    preds = []
    bert_score_list = {"bert_P": [], "bert_R": [], "bert_F1": []}
    for i in range(len(texts)):
        if reftext is not None:
            refs.append(reftext)
            preds.append(texts[i])
        else:
            for j in range(i+1, len(texts)):
                refs.append(texts[j])
                preds.append(texts[i])
    P, R, F1 = score(refs, preds, lang="en", verbose=False)
    if rouge_topk > 0:
        Plist = 1 - P.topk(rouge_topk)[0].mean()
        Rlist = 1 - R.topk(rouge_topk)[0].mean()
        F1list = 1 - F1.topk(rouge_topk)[0].mean()
    else:
        Plist = 1 - P.mean()
        Rlist = 1 - R.mean()
        F1list = 1 - F1.mean()
    bert_score_list["bert_P"] = P.tolist()
    bert_score_list["bert_R"] = R.tolist()
    bert_score_list["bert_F1"] = F1.tolist()
    return Plist, Rlist, F1list, bert_score_list


def get_tpr_at_fpr(y_true, y_scores, target_fpr=0.05):
    # Get the FPR and TPR at various thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the index where FPR is closest to or just under the target
    idx = np.where(fpr <= target_fpr)[0][-1]
    return tpr[idx]


with open(infile) as fin:
    data = json.load(fin)

ref_all_metrics = {}

labels = []
all_metrics = {}
all_metrics_audio = {}

for idx, datapiece in enumerate(tqdm(data)):
    labels.append(datapiece["label"])
    datapiece["precompute"] = {}
    for metric, value in datapiece["pred"].items():
        key = datapiece["audio"] + datapiece["question"]
        if metric == "pred":
            reftext = datapiece["answer"]
            rouge_scores, rouge_scorelist = compute_rouge(value, reftext=reftext)
            datapiece["precompute"]["rouge_1_reftext"] = rouge_scorelist["rouge_1"]
            datapiece["precompute"]["rouge_2_reftext"] = rouge_scorelist["rouge_2"]
            datapiece["precompute"]["rouge_l_reftext"] = rouge_scorelist["rouge_l"]

            rouge_scores, rouge_scorelist = compute_rouge(value, reftext=None)
            datapiece["precompute"]["rouge_1_predtext"] = rouge_scorelist["rouge_1"]
            datapiece["precompute"]["rouge_2_predtext"] = rouge_scorelist["rouge_2"]
            datapiece["precompute"]["rouge_l_predtext"] = rouge_scorelist["rouge_l"]

            bert_P, bert_R, bert_F1, bert_score_list = get_bert_score(value, reftext=reftext)
            datapiece["precompute"]["bert_P_reftext"] = bert_score_list["bert_P"]
            datapiece["precompute"]["bert_R_reftext"] = bert_score_list["bert_R"]
            datapiece["precompute"]["bert_F1_reftext"] = bert_score_list["bert_F1"]

            bert_P, bert_R, bert_F1, bert_score_list = get_bert_score(value, reftext=None)
            datapiece["precompute"]["bert_P_predtext"] = bert_score_list["bert_P"]
            datapiece["precompute"]["bert_R_predtext"] = bert_score_list["bert_R"]
            datapiece["precompute"]["bert_F1_predtext"] = bert_score_list["bert_F1"]

with open(outfile, "w") as fout:
    json.dump(data, fout, indent=4)