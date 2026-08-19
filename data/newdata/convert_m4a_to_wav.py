import os
import json
from tqdm import tqdm
import argparse
import subprocess
from pathlib import Path


def m4a_to_wav(input_path, output_path=None, sample_rate=16000):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".wav")

    subprocess.run(
        [
            "ffmpeg",
            "-y",                 # overwrite output without asking
            "-i", str(input_path),
            "-ar", str(sample_rate),  # resample to 16 kHz
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path

list_of_audios = []
for filename in tqdm(os.listdir("miavideos_flat")):
    if ".m4a" in filename:
        filepath = os.path.join("miavideos_flat", filename)
        outpath = os.path.join("miavideos_wav", filename.replace(".m4a", ".wav"))
        outpath = m4a_to_wav(filepath, outpath)
        list_of_audios.append(outpath)

with open("list_of_audios.json", "w") as fout:
    json.dump(list_of_audios, fout, indent=4)