import json
import os
import random


caption_prompts = [
    "Describe the audio in detail, pay attention to speech content, speaker attributes and styles, as well as audio events.",
    'Provide a detailed description of the audio, focusing on speech content, speaker attributes, speaking style, and audio events.',
    'Describe the audio thoroughly, paying attention to what is said, speaker characteristics, speaking styles, and any sound events.',
    'Give a detailed account of the audio, including the speech content, speaker traits, vocal style, and notable audio events.',
    'Analyze the audio in detail, covering spoken content, speaker identity and style, and relevant acoustic events.',
    'Write a detailed description of the audio that includes speech content, speaker attributes, stylistic delivery, and audio events.',
    'Describe the audio with attention to the spoken words, speaker qualities, speaking style, and any notable sounds or events.',
    'Provide a comprehensive description of the audio, noting speech content, speaker characteristics, style, and audio events.',
    'Explain the audio in detail, focusing on the speech, the speakers, their style, and any audio events present.',
    'Offer a detailed audio description that captures speech content, speaker attributes, speaking styles, and audio events.',
    'Describe the audio in depth, including what is spoken, speaker features, vocal style, and any significant audio events.',
    'Provide a detailed summary of the audio, paying close attention to speech content, speaker attributes, style, and audio events.',
    'Characterize the audio in detail, covering spoken content, speaker characteristics, delivery style, and audio events.',
    'Describe the audio carefully, with emphasis on speech content, speaker properties, stylistic aspects, and audio events.',
    'Create a detailed description of the audio, including the words spoken, speaker attributes, style, and notable audio events.',
    'Detail the audio content, paying attention to speech, speaker attributes, speaking styles, and any audio events.',
    'Describe the audio comprehensively, noting the speech content, speaker traits, stylistic features, and audio events.',
    'Provide an in-depth description of the audio that addresses speech content, speaker attributes, vocal styles, and audio events.',
    'Describe the audio carefully, focusing on spoken content, speaker identity or attributes, delivery style, and audio events.',
    'Give a thorough audio description, including speech content, speaker characteristics, speaking style, and relevant audio events.',
]

with open("allmia_data_with_question_gemini3.json") as fin:
    data = json.load(fin)

with open("allmia_data_category.json") as fin:
    category = json.load(fin)

audio_to_cat = {}
for datapiece in category:
    audio_to_cat[datapiece["audio"]] = datapiece["language"]

english_data = []
for datapiece in data:
    if audio_to_cat[datapiece["audio"]] == "English":
        english_data.append(datapiece)

print(len(english_data))
random.shuffle(english_data)
train_audio = english_data[:640]
holdout_audio = english_data[640:]
traindata_asrcap = []
traindata_qa = []
for datapiece in train_audio:
    newpiece = {
        "audio": datapiece["audio"],
        "question": "Transcribe the audio to text",
        "answer": datapiece["transcription"]
    }
    traindata_asrcap.append(newpiece)
    newpiece = {
        "audio": datapiece["audio"],
        "question": random.choice(caption_prompts),
        "answer": datapiece["caption"]
    }
    traindata_asrcap.append(newpiece)

    for question in datapiece["questions"]:
        if "question" not in question or "answer" not in question:
            continue
        newpiece = {
            "audio": datapiece["audio"],
            "question": question["question"],
            "answer": question["answer"]
        }
        traindata_qa.append(newpiece)

with open("newdata_train_capasr.json", "w") as fout:
    json.dump(traindata_asrcap, fout, indent=4, ensure_ascii=False)

with open("newdata_train_QA.json", "w") as fout:
    json.dump(traindata_qa, fout, indent=4, ensure_ascii=False)

with open("newdata_train_capasr_QA.json", "w") as fout:
    json.dump(traindata_qa+traindata_asrcap, fout, indent=4, ensure_ascii=False)

audio_split = {
    "train": [audio["audio"] for audio in train_audio],
    "test": [audio["audio"] for audio in holdout_audio],
}

with open("audio_split.json", "w") as fout:
    json.dump(audio_split, fout, indent=4)