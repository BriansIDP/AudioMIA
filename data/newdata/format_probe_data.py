import json
import random
import os


letters = ["A", "B", "C", "D", "E", "F"]

template = """You will hear a short part of an audio clip.

You have heard the complete audio before or know the content of this audio, then use your knowledge to answer the following question.

Question: {}
"""

with open("../per_audio_testset.json") as fin:
    audiodata = json.load(fin)

audio_to_label = {}
trainsetaudios = []
for datapiece in audiodata:
    audioname = datapiece["audio"].split("/")[-1]
    audio_to_label[audioname] = datapiece["label"]
    trainsetaudios.append(audioname)

with open("filtered_audio_data_probe.json") as fin:
    testset = json.load(fin)

traindata = []
shortdata = []
evaldata = []
for datapiece in testset:
    audioname = datapiece["audio"].split("/")[-1]
    for question in datapiece["questions"]:
        q_format = template.format(question["question"])
        newpiece = {
            "audio": datapiece["audio"],
            "type": "probe",
            "question": q_format,
            "answer": question["answer"],
        }
        if question["answer"] not in question["options"]:
            options = question["options"] + [question["answer"]]
        else:
            options = question["options"]
        random.shuffle(options)
        answer = letters[options.index(question["answer"])]
        choices = ["{}. {}".format(letters[i], value) for i, value in enumerate(options)]
        mcq_format = "{}\nChoose from:\n{}".format(question["question"], "\n".join(choices))
        newpiece_2 = {
            "audio": os.path.join("/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/newdata", datapiece["audio"]),
            "type": "probe",
            "question": mcq_format,
            "answer": answer,
        }
        # if audioname in audio_to_label and audio_to_label[audioname] == 0:
        shortdata.append(newpiece)
        traindata.append(newpiece_2)
        # elif audioname in audio_to_label and audio_to_label[audioname] == 1:
        #     evaldata.append(newpiece_2)

with open("testset_probe.json", "w") as fout:
    json.dump(traindata, fout, indent=4)