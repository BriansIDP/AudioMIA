import json


with open("per_audio_testset.json") as fin:
    data = json.load(fin)

pos_data = []
neg_data = []
for datapiece in data:
    if datapiece["label"] == 1:
        pos_data.append(datapiece)
    else:
        neg_data.append(datapiece)

with open("per_audio_testset_small.json", "w") as fout:
    json.dump(pos_data[:500] + neg_data[:500], fout, indent=4)