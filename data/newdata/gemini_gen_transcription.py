import time
import copy
import openai
import os
import base64
from mimetypes import guess_type
import mimetypes
import json
import concurrent.futures
import threading
import subprocess
import tempfile
from tqdm import tqdm
import regex


pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

system_prompt = """Transcribe the audio content in the given audio file into text

Output format:
{
  "transcription": "the transcription of the audio"
}
"""

system_prompt_cap = """You are given an audio. Your task is to generate a detailed description about the audio, pay attention to both the speech content and the speaker characteristics. Output the description in a narrative way. The description should be as detailed as possible.

Output format:
{{
    "description": "The generated caption of the audio",
}}
"""

def local_file_to_data_url(file_path: str) -> str:
    """
    Convert a local file to a data: URL.
    Ensures .wav files use the standard 'audio/wav' MIME type.
    """
    # Make sure .wav is registered and normalize variants
    mimetypes.init()
    mimetypes.add_type('audio/wav', '.wav')  # idempotent

    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    # Normalize WAV MIME quirks across platforms
    if ext == '.wav' and mime_type in (None, 'audio/x-wav', 'audio/wave', 'audio/vnd.wave'):
        mime_type = 'audio/wav'

    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(file_path, "rb") as fp:
        base64_encoded_data = base64.b64encode(fp.read()).decode("ascii")

    return f"data:{mime_type};base64,{base64_encoded_data}"

def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")

def gpt4_caption(audiopath):
    # params:
    # texts:
    # type: list
    # content: each input text as a element
    client = openai.AzureOpenAI(
        azure_endpoint="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
        api_version="2024-03-01-preview",
        api_key="Aw7Q1fCVT1g9mFGMPOplMhVUZWXWNtrl_GPT_AK",
    )
    # client = openai.AzureOpenAI(
    #     azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/v2/crawl",
    #     api_version="2024-03-01-preview",
    #     api_key="KNvcapi3WFBve5HsLONHacRY5rlWHrO3"
    # )
    url1 = local_file_to_data_url(audiopath)
    msg=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url1}},
                {"type": "text", "text": system_prompt_cap},
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="gemini-3.5-flash", # "gpt-4-turbo-2024-04-09", # "gpt-4o-2024-05-13", # 
        messages=msg,
        seed=2025
    )
    content = completion.choices[0].message.content
    json_obj = pattern.findall(content)
    pred_answer = json.loads(json_obj[0])
    return pred_answer

json_file = "eval_audios_filter.json"
output_json = "eval_audios_filter_cap.json"


with open(json_file, 'r') as fp:
    data = json.load(fp)

data = [{"audio": audio} for audio in data]

def gpt_extract(item):
    try:
        res = gpt4_caption(item["audio"])
        # item["transcription"] = res["transcription"]
        item["description"] = res["description"]
        return item
    except Exception as e:
        # raise e
        return item

# data = data[:10]
# gpt_extract(data[0])
# exit()
print(len(data))
# task_video_pairs = task_video_pairs[:5]

# with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
#     responses = list(tqdm(executor.map(gpt_extract, data)))

result = [r for r in data if "description" in r]
ignore = [r for r in data if "description" not in r]

print(output_json)
print(len(result), len(ignore))
k = 0

while len(ignore) != 0 and k < 20:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        responses = list(tqdm(executor.map(gpt_extract, ignore)))
    result += [r for r in responses if "description" in r]
    ignore = [r for r in responses if "description" not in r]
    with open(output_json, 'w') as fp:
        json.dump(result, fp, indent=4)
    print(output_json)
    print(len(result), len(ignore))

    k += 1