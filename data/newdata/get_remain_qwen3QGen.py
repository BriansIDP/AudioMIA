import json


with open("audio_split.json") as fin:
    data = json.load(fin)

all_audios = set()
for setname, audios in data.items():
    for audio in audios:
        all_audios.add(audio)

with open("allmia_data_with_question_qwen3omni_0_200.json") as fin:
    data = json.load(fin)
with open("allmia_data_with_question_qwen3omni_200_400.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_400_600.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_600_800.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_800_1000.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_1000_1300.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_remain_0_50.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_remain_50_100.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_remain_100_150.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_remain_150_200.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_remain_200_250.json") as fin:
    data += json.load(fin)
with open("allmia_data_with_question_qwen3omni_remain_250_300.json") as fin:
    data += json.load(fin)

audio_to_datapiece = {}
with open("allmia_data.json") as fin:
    origdata = json.load(fin)
for datapiece in origdata:
    audio_to_datapiece[datapiece["audio"]] = datapiece

for datapiece in data:
    if datapiece["audio"] in all_audios:
        all_audios.remove(datapiece["audio"])

remain_data = []
for audio in all_audios:
    remain_data.append(audio_to_datapiece[audio])

with open("allmia_data_with_question_qwen3omni.json", "w") as fout:
    json.dump(data, fout, indent=4)