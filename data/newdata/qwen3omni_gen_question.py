
import librosa
import audioread

import os, sys
import torch
import warnings
import numpy as np
import json
from tqdm import tqdm
import regex
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

summarize_prompt = '''
### Task:
You are given an audio. Your task is to generate questions that test how much a specific model memorizes what has been said or can be heard in this audio.

The question cannot be answered with commonsense knowledge or some reasoning without listening to the audio.

Good example questions include but not limited to:
What was said immediately before the sentence "xxx"
What is the most important factor to contribute to the second goal mentioned in the audio? (The second goal can only be inferred from the audio)

Bad question examples:
According to Kate's explanation, what is a key requirement for the schools to merge? (This can be inferred by reasoning through the options, so this is a bad example)
Why did the Mauritius tax authorities deny the taxpayer's claim for the 80% partial exemption? (This can also be inferred by using commonsense knowledge from the options)

Generate 6-10 questions for each audio. The questions should try the best to cover the following aspects:
1. speech content
2. speakers, their characteristics, their talking styles and their emotions
3. Any obvious audio events

Output format:
{
    "results": [
        {
            "question": "Generated question about the speech.",
            "answer": "Answer to the question.",
        },
        {
            "question": "Generated question about the speech.",
            "answer": "Answer to the question.",
        },
        ...
    ]
}
'''


TRANSFORMERS_USE_FLASH_ATTN2 = True


def _load_model_processor():
    from transformers import Qwen3OmniMoeForConditionalGeneration
    if TRANSFORMERS_USE_FLASH_ATTN2:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH,
                                                                        dtype='auto',
                                                                        attn_implementation='flash_attention_2')
    else:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, dtype='auto')

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    model.to("cuda")
    return model, processor

def run_model(model, processor, messages, return_audio, use_audio_in_video):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)
    response = model.generate(**inputs, thinker_return_dict_in_generate=True, thinker_max_new_tokens=8192, thinker_do_sample=True, use_audio_in_video=use_audio_in_video, return_audio=False)
    response = processor.batch_decode(response.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response
    

if __name__ == "__main__":
    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    model, processor = _load_model_processor()

    with open("all_miadata_qwen3gen_remain.json") as fin:
        data = json.load(fin)

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    finished = []
    unfinished = data[start:end]
    while unfinished != []:
        new_unfinished = []
        for datapiece in tqdm(unfinished):
            audio_path = datapiece["audio"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": summarize_prompt},
                    ]
                }
            ]

            USE_AUDIO_IN_VIDEO = True
            RETURN_AUDIO = False
            try:
                response = run_model(model=model, messages=messages, processor=processor, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                json_obj = pattern.findall(response)
                questions = json.loads(json_obj[0])["results"]
                datapiece["questions"] = questions
                finished.append(datapiece)
            except:
                new_unfinished.append(datapiece)
        print("Finished: {}\tUnfinished: {}".format(len(finished), len(new_unfinished)))
        unfinished = new_unfinished

        with open("allmia_data_with_question_qwen3omni_remain_{}_{}.json".format(start, end), "w") as fout:
            json.dump(finished, fout, indent=4)
