import json


with open("audio_split.json") as fin:
    split = json.load(fin)

audio_to_split = {}
for setname, audios in split.items():
    for audio in audios:
        audio_to_split[audio] = setname

with open("allmia_data_with_question_gemini3.json") as fin:
    data = json.load(fin)

alldata = []
traindata_qa = []
for datapiece in data:
    if datapiece["audio"] not in audio_to_split:
        continue
    setname = audio_to_split[datapiece["audio"]]
    newpiece = {
        "audio": datapiece["audio"],
        "question": "Transcribe the audio to text",
        "answer": datapiece["transcription"],
        "label": 0 if setname == "train" else 1
    }
    alldata.append(newpiece)
    newpiece = {
        "audio": datapiece["audio"],
        "question": "Describe the audio in detail",
        "answer": datapiece["caption"],
        "label": 0 if setname == "train" else 1
    }
    alldata.append(newpiece)

    for question in datapiece["questions"]:
        if not isinstance(question, dict):
            continue
        newpiece = {
            "audio": datapiece["audio"],
            "question": question["question"],
            "answer": question["answer"],
            "label": 0 if setname == "train" else 1
        }
        traindata_qa.append(newpiece)

with open("testdata_QA_train.json", "w") as fout:
    json.dump(traindata_qa, fout, indent=4)