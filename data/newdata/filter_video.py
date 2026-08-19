import json
import os
from tqdm import tqdm


with open("filtered_urls.json") as fin:
    data = json.load(fin)
vid_to_meta = {}
covered_title = set()
for datapiece in tqdm(data):
    vid = datapiece["url"].replace("https://www.youtube.com/watch?v=", "")
    audio_path = os.path.join("miavideos_wav", "{}.wav".format(vid))
    if os.path.exists(audio_path) and datapiece["title"] not in covered_title:
        vid_to_meta[vid] = {
            "video_id": vid,
            "title": datapiece["title"],
            "duration": datapiece["duration"],
            "upload_date": datapiece["upload_date"]
        }
        covered_title.add(datapiece["title"])

retain_category = []
with open("category_queries.json") as fin:
    data = json.load(fin)
for cat, query in data.items():
    retain_category.extend(query)

with open("youtube_urls_ytdlp.json") as fin:
    data = json.load(fin)

for datapiece in data:
    vid = datapiece["url"].replace("https://www.youtube.com/watch?v=", "")
    if vid in vid_to_meta:
        vid_to_meta[vid]["category"] = datapiece["category_name"]
        vid_to_meta[vid]["query"] = datapiece["query"].replace("after:2026-01-01", "").strip()

vid_to_meta_filter = {}
duration_dist = []
for vid, meta in vid_to_meta.items():
    if meta["query"] in retain_category:
        vid_to_meta_filter[vid] = meta
        duration_dist.append(meta["duration"])

print("duration mean: {}".format(sum(duration_dist)/len(duration_dist)))

with open("video_to_metadata.json", "w") as fout:
    json.dump(vid_to_meta_filter, fout, indent=4)