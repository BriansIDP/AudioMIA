import json


with open("/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/exp/qwen25_omni_sft_newQA_r1024_a2048_probe_ref/mia_qwen25omni_generation_1_samples_minkpp_completion_probe_newdata.json") as fin:
    data = json.load(fin)

audio_to_acc = {}
for datapiece in data:
    audiopath = datapiece["audio"] #.replace("/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/newdata", "")
    if audiopath not in audio_to_acc:
        audio_to_acc[audiopath] = []
    if datapiece["answer"][0] == datapiece["pred"]["pred"][0][0]:
        audio_to_acc[audiopath].append(1)
    else:
        audio_to_acc[audiopath].append(0)

filtered_train = set()
filtered_test = set()
threshold = 1.0
for audiopath, hits in audio_to_acc.items():
    acc_audio = sum(hits) / len(hits)
    if acc_audio < threshold and "miavideos_wav" in audiopath:
        filtered_test.add(audiopath)
    elif acc_audio < threshold and "miavideos_short" in audiopath and "_001" in audiopath:
        filtered_test.add(audiopath)

# with open("train_audios.json") as fin:
#     trainaudios = json.load(fin)

# with open("eval_audios.json") as fin:
#     evalaudios = json.load(fin)

# filtered_train = set()
# for audio in trainaudios:
#     if audio in audio_to_acc:
#         acc_audio = sum(audio_to_acc[audio]) / len(audio_to_acc[audio])
#         if acc_audio < 0.5:
#             filtered_train.add(audio)

# filtered_eval = set()
# for audio in evalaudios:
#     if audio in audio_to_acc:
#         acc_audio = sum(audio_to_acc[audio]) / len(audio_to_acc[audio])
#         if acc_audio < 0.5:
#             filtered_eval.add(audio)

print(len(filtered_test))

# with open("train_audios_full.json", "w") as fout:
#     json.dump(list(filtered_train), fout, indent=4)

with open("eval_audios_filter.json", "w") as fout:
    json.dump(list(filtered_test), fout, indent=4)