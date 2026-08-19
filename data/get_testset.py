import json
import random


with open("train_data_qa.json") as fin:
    data = json.load(fin)
random.shuffle(data)

with open("eval_data_qa.json") as fin:
    testdata = json.load(fin)
random.shuffle(testdata)

audio_to_question_train = {}
audio_to_question_test = {}
with open("per_audio_testset_small.json") as fin:
    audio_split = json.load(fin)
audio_split = audio_split[:100] + audio_split[-100:]

with open("all_mia_data_QA.json") as fin:
    subqa = json.load(fin)

audio_to_qa = {}
for datapiece in subqa:
    q_data = []
    audiopath = datapiece["audio"].split("/")[-1]
    for qa in datapiece["questions"]:
        newpiece = {
            "audio": datapiece["audio"],
            "question": qa["question"],
            "answer": qa["answer"],
        }
        q_data.append(newpiece)
    audio_to_qa[audiopath] = q_data

newdata = []
# for datapiece in data:
#     datapiece["label"] = 0
#     audiopath = datapiece["audio"].split("/")[-1]
#     if audiopath not in audio_to_question_train:
#         audio_to_question_train[audiopath] = []
#     audio_to_question_train[audiopath].append(datapiece)

# for datapiece in testdata:
#     datapiece["label"] = 1
#     audiopath = datapiece["audio"].split("/")[-1]
#     if audiopath not in audio_to_question_test:
#         audio_to_question_test[audiopath] = []
#     audio_to_question_test[audiopath].append(datapiece)

for audiodata in audio_split:
    audiopath = audiodata["audio"].split("/")[-1]
    for datapiece in audio_to_qa[audiopath]:
        datapiece["label"] = audiodata["label"]
    newdata.extend(audio_to_qa[audiopath])
    # if audiodata["label"] == 0:
    #     newdata.extend(audio_to_question_train[audiopath])
    # else:
    #     newdata.extend(audio_to_question_test[audiopath])

with open("per_audio_all_question_testset_small.json", "w") as fout:
    json.dump(newdata, fout, indent=4)