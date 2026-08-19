import json
import random


with open("all_mia_data_Caption.json") as fin:
    cap_data = json.load(fin)

with open("all_mia_data_QA.json") as fin:
    qa_data = json.load(fin)

# all_audios = []
# for datapiece in cap_data:
#     if datapiece["audio"] not in all_audios:
#         all_audios.append(datapiece["audio"])
# random.shuffle(all_audios)
# train_audios = all_audios[:1800]
# eval_audios = all_audios[1800:]

with open("train_audios.json") as fout:
    # json.dump(train_audios, fout, indent=4, ensure_ascii=False)
    train_audios = json.load(fout)
with open("eval_audios.json") as fout:
    # json.dump(eval_audios, fout, indent=4, ensure_ascii=False)
    eval_audios = json.load(fout)

traindata = []
evaldata = []
for datapiece in cap_data:
    transcript = datapiece["transcription"]
    newpiece = {
        "audio": "data/" + datapiece["audio"],
        "question": "Transcribe the speech content and output the transcription.",
        "answer": transcript,
    }
    # newpiece2 = {
    #     "audio": "data/" + datapiece["audio"],
    #     "question": "Describe the audio content in detail.",
    #     "answer": datapiece["caption"],
    # }
    if datapiece["audio"] in train_audios:
        newpiece["label"] = 0
        traindata.append(newpiece)
        # traindata.append(newpiece2)
    elif datapiece["audio"] in eval_audios:
        newpiece["label"] = 1
        evaldata.append(newpiece)
        # evaldata.append(newpiece2)

# train_qa_test = []
# eval_qa_test = []
# for datapiece in qa_data:
#     for question in datapiece["questions"]:
#         newpiece = {
#             "audio": "data/" + datapiece["audio"],
#             "question": question["question"],
#             "answer": question["answer"],
#         }
#         if datapiece["audio"] in train_audios:
#             train_qa_test.append(newpiece)
#         elif datapiece["audio"] in eval_audios:
#             eval_qa_test.append(newpiece)

# random.shuffle(traindata)
# with open("train_data_caption.json", "w") as fout:
#     json.dump(traindata, fout, indent=4, ensure_ascii=False)
# with open("valid_data.json", "w") as fout:
#     json.dump(traindata[:500], fout, indent=4, ensure_ascii=False)
# with open("eval_data_caption.json", "w") as fout:
#     json.dump(evaldata, fout, indent=4, ensure_ascii=False)
# with open("train_data_qa.json", "w") as fout:
#     json.dump(train_qa_test, fout, indent=4, ensure_ascii=False)
# with open("eval_data_qa.json", "w") as fout:
#     json.dump(eval_qa_test, fout, indent=4, ensure_ascii=False)
# with open("train_data.json", "w") as fout:
#     json.dump(traindata+train_qa_test, fout, indent=4, ensure_ascii=False)
# with open("eval_data.json", "w") as fout:
#     json.dump(eval_qa_test+evaldata, fout, indent=4, ensure_ascii=False)

with open("all_mia_data_transcription.json", "w") as fout:
    json.dump(traindata+evaldata, fout, indent=4, ensure_ascii=False)