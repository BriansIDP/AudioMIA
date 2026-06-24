import json


template = """You will hear a short public speech clip.

Use the audio only to infer who the speaker is, and then answer the following question from your knowledge about this speaker.

Question: {}
"""

with open("train_data_future_question.json") as fin:
    data = json.load(fin)

with open("train_data_future_question_short.json") as fin:
    shortdata = json.load(fin)

all_audios = set()
for datapiece in data[:750]:
    all_audios.add(datapiece["audio"])

traindata = []
validdata = []
for datapiece in data + shortdata:
    for question in datapiece["questions"]:
        q_format = template.format(question["question"])
        newpiece = {
            "audio": datapiece["audio"],
            "type": "probe",
            "question": q_format,
            "answer": question["answer"]
        }
        if datapiece["audio"] in all_audios:
            traindata.append(newpiece)
        else:
            if len(question["answer"].split()) < 5:
                validdata.append(newpiece)

with open("train_probe.json", "w") as fout:
    json.dump(traindata, fout, indent=4)

with open("valid_probe.json", "w") as fout:
    json.dump(validdata, fout, indent=4)

with open("/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/train_data.json") as fin:
    origdata = json.load(fin)

with open("train_probe_full.json", "w") as fout:
    json.dump(traindata+origdata, fout, indent=4)