import openai
import base64
from mimetypes import guess_type
import json
from tqdm import tqdm
import os
import regex
import concurrent.futures
import threading
import subprocess
import tempfile
# from moviepy import VideoFileClip


pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

summarize_prompt = '''
### Task:
You are given an audio and the transcription of the speech content. Your task is to generate questions that test how much a specific model memorizes what this speaker has said.

Assume you will provide the model with another audio of the same speaker, and then you should ask questions about the information in the transcription below. The question should be about the information after the first 10 seconds of the audio.

Example questions include but not limited to:
What has this speaker said about <something>?
When did this speaker say about going to the hospital?
What software did this speaker mention?

Transcription:
{}

Generate 5-10 questions for each audio.

Output format:
{{
    "results": [
        {{
            "question": "Generated question about the speech.",
            "answer": "Answer to the question."
        }},
        {{
            "question": "Generated question about the speech.",
            "answer": "Answer to the question."
        }},
        ...
    ]
}}
'''

LOCK = threading.Lock()

json_file = "/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/data/train_data_asr.json"
output_json = "train_data_future_question.json"
temp_json = "tmp.json"

ff = open(temp_json, 'w')


import os
import base64
import mimetypes

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

def gemini_caption(audio_path_1, transcript):
    client = openai.AzureOpenAI(
        azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/v2/crawl",
        api_version="2024-03-01-preview",
        api_key="KNvcapi3WFBve5HsLONHacRY5rlWHrO3"
    )
    url1 = local_file_to_data_url(audio_path_1)
    # url2 = local_file_to_data_url(audio_path_2)
    completion = client.chat.completions.create(
        model="gemini-2.5-pro-preview-06-05",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url1}},
                # {"type": "image_url", "image_url": {"url": url2}},
                {"type": "text", "text": summarize_prompt.format(transcript)},
            ]
        }],
        seed=2027
    )
    content = completion.choices[0].message.content
    json_obj = pattern.findall(content)
    to_return = json.loads(json_obj[0])["results"]
    return to_return

def gemini_extract(item):
    try:
        audiopath = os.path.join("/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/AudioMIA/", item["audio"])
        res = gemini_caption(audiopath, item["answer"])
        item["questions"] = res
        with LOCK:
            json_str = json.dumps(item)
            ff.write(json_str + "\n")
        return item
    except Exception as e:
        # print(e)
        return item

with open(json_file, 'r') as fp:
    data = json.load(fp)

result = []
covered = set()
# if os.path.exists(output_json):
#     with open(output_json, 'r') as fp:
#         resultdata = json.load(fp)
#     for datapiece in resultdata:
#         result.append(datapiece)
#         covered.add(datapiece["audio"])

ignore = []
for item in data:
    # if item["audio"] not in covered:
    if len(item["answer"].split()) > 100:
        ignore.append(item)
# ignore = ignore[:20]
# gemini_extract(ignore[0])
# ignore = ignore[:10]
# exit()

print(len(result), len(ignore))
k = 0
while len(ignore) != 0 and k < 200:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        responses = list(tqdm(executor.map(gemini_extract, ignore)))
    ignore = []
    for r in responses:
        if "questions" in r:
            result.append(r)
        else:
            ignore.append(r)
    with open(output_json, 'w') as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)
    print(output_json)
    print(len(result), len(ignore))
    k += 1