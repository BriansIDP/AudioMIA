import json
import os
import argparse
import re
import subprocess
import sys
import copy
from tqdm import tqdm
from pathlib import Path


# Chunk constraints, in seconds
MIN_SEC = 120.0      # 2 minutes
MAX_SEC = 180.0      # 3 minutes
TARGET_SEC = 150.0   # aim for ~2.5 minutes when possible

# Silence detection tuning
NOISE_DB = -35       # silence threshold in dB; increase if it misses pauses
MIN_SILENCE_SEC = 0.5  # minimum pause length to count as silence


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def get_silence_midpoints(path: Path, duration: float):
    """
    Use ffmpeg silencedetect to find silent regions.
    Returns midpoints of silence intervals as candidate cut positions.
    """
    af = f"silencedetect=noise={NOISE_DB}dB:d={MIN_SILENCE_SEC}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i", str(path),
        "-af", af,
        "-f", "null",
        "-",
    ]

    p = run(cmd)
    if p.returncode != 0:
        sys.exit(f"ffmpeg silencedetect failed: {p.stderr}")

    start_re = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
    end_re = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")

    points = []
    last_start = None

    for line in p.stderr.splitlines():
        m = start_re.search(line)
        if m:
            last_start = float(m.group(1))
            continue

        m = end_re.search(line)
        if m and last_start is not None:
            end = float(m.group(1))
            if end > last_start:
                points.append((last_start + end) / 2.0)
            last_start = None

    # If ffmpeg reported a final silence with no end, treat it as ending at EOF.
    if last_start is not None and duration > last_start:
        points.append((last_start + duration) / 2.0)

    return sorted(points)


def choose_chunk_count(duration: float):
    """
    Choose a number of chunks so that average chunk length is between 2 and 3 minutes.
    Prefer chunk lengths close to TARGET_SEC.
    """
    best_n = None
    best_err = None

    max_n = int(duration / MIN_SEC) + 2

    for n in range(1, max_n + 1):
        avg = duration / n
        if MIN_SEC <= avg <= MAX_SEC:
            err = abs(avg - TARGET_SEC)
            if best_err is None or err < best_err:
                best_err = err
                best_n = n

    return best_n


def choose_cut_points(duration: float, silence_points):
    """
    Decide where to cut.

    Returns a list of end times:
        [end_of_chunk_1, end_of_chunk_2, ..., duration]
    """
    n = choose_chunk_count(duration)

    # If no valid 2-3 minute split exists, do nothing.
    # Example: a 3:30 file cannot be split into two chunks both >= 2:00.
    if not n or n <= 1:
        return []

    cuts = []
    current = 0.0

    for i in range(1, n):
        remaining_chunks = n - i

        # Keep current chunk between MIN_SEC and MAX_SEC,
        # while also leaving enough time for remaining chunks.
        low = max(
            current + MIN_SEC,
            duration - remaining_chunks * MAX_SEC,
        )
        high = min(
            current + MAX_SEC,
            duration - remaining_chunks * MIN_SEC,
        )

        desired = current + (duration - current) / (remaining_chunks + 1)

        if low <= high:
            candidates = [p for p in silence_points if low <= p <= high]

            if candidates:
                # Choose the silence closest to the ideal target point.
                cut = min(candidates, key=lambda p: abs(p - desired))
            else:
                # No silence found in the valid window; cut at the best valid time.
                cut = min(max(desired, low), high)
        else:
            # Safety fallback; normally should not happen if choose_chunk_count worked.
            cut = min(max(desired, current + 1.0), duration - 1.0)

        cuts.append(cut)
        current = cut

    cuts.append(duration)
    return cuts


def export_chunks(path: Path, cuts, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower() or ".wav"
    starts = [0.0] + cuts[:-1]
    created = []

    for idx, (start, end) in enumerate(zip(starts, cuts), start=1):
        out = outdir / f"{path.stem}_{idx:03d}{ext}"
        length = end - start

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(path),
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-map", "0:a:0",
            "-c:a", "copy",  # fast; remove this line if your codec/container complains
            str(out),
        ]

        p = subprocess.run(cmd)
        if p.returncode != 0:
            sys.exit(f"ffmpeg failed while writing {out}")

        created.append(out)

    return created


if __name__ == "__main__":
    with open("video_to_metadata.json") as fin:
        data = json.load(fin)

    categories = [
        "Academic Conference Presentations",
        "Independent & Professional Podcasts",
        "News Broadcasts & Interviews",
        "Mental Health Vlogs & Personal Video Diaries",
        "Corporate Earnings Calls & Business Webinars",
        "Legal Analysis & Case Summaries",
    ]

    shortvideos = []
    cutvideos = []
    for vid, datapiece in data.items():
        if datapiece["duration"] < 200 and datapiece["category"] in categories:
            datapiece["audio"] = os.path.join("miavideos_wav", datapiece["video_id"]+".wav")
            shortvideos.append(datapiece)
        elif datapiece["duration"] < 600 and datapiece["category"] in categories:
            cutvideos.append(datapiece)

    print(len(shortvideos))
    print(len(cutvideos))
    newdata = []

    for video in tqdm(cutvideos):
        vidpath = os.path.join("miavideos_wav", video["video_id"]+".wav")
        silence_points = get_silence_midpoints(vidpath, video["duration"])
        cuts = choose_cut_points(video["duration"], silence_points)
        if cuts:
            files = export_chunks(Path(vidpath), cuts, Path("miavideos_short"))
            timestamps = [0.0] + cuts
            for idx, file in enumerate(files):
                newpiece = copy.deepcopy(video)
                newpiece["audio"] = str(file)
                newpiece["duration"] = timestamps[idx+1] - timestamps[idx]
                newdata.append(newpiece)

    with open("filtered_audio_data.json", "w") as fout:
        json.dump(shortvideos + newdata, fout, indent=4)