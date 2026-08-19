import json

with open("per_audio_all_question_testset_small.json") as fin:
    data = json.load(fin)

with open("all_mia_data_Caption.json") as fin:
    captions = json.load(fin)
allcaptions = {}
for datapiece in captions:
    allcaptions[datapiece["audio"].split("/")[-1]] = datapiece["caption"]

allaudios = set()
capdata = []
for datapiece in data:
    if datapiece["audio"] not in allaudios:
        allaudios.add(datapiece["audio"])
        audiokey = datapiece["audio"].split("/")[-1]
        newpiece = {"audio": datapiece["audio"], "question": "Describe the audio in detail.", "answer": allcaptions[audiokey], "label": datapiece["label"]}
        capdata.append(newpiece)

with open("per_audio_testset_small.json", "w") as fout:
    json.dump(capdata, fout, indent=4)
