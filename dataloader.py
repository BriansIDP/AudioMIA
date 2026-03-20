#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,  # not used, but handy for quick tests
)
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.nn.utils.rnn import pad_sequence


# -----------------------------
# Dataset
# -----------------------------
class ShadowDataset(Dataset):
    """
    Loads either:
      - Local .jsonl or .json file (list or newline records)
      - HF dataset (path on the Hub) via datasets.load_dataset
    Produces dicts with tokenized input_ids/attention_mask/labels where only the
    FINAL assistant reply in the sample is labeled.
    """

    def __init__(
        self,
        source: str,
        tokenizer: AutoTokenizer,
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.max_len = 8192
        with open(source) as fin:
            data = json.load(fin)
        self.data = []
        for datapiece in data:
            self.data.append(self.preprocess(datapiece))

    def preprocess(self, item):
        audiopath = item["audio"]
        question = item["question"]
        answer = item["answer"]
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audiopath},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
        return conversation

    def _build_prompt_and_full_text(self, msgs: List[Dict[str, str]], tokenizer) -> (str, str):
        # ensure there's at least a terminal assistant; if not, synthesize empty assistant
        last_is_assistant = (len(msgs) > 0 and msgs[-1]["role"] == "assistant")
        if not last_is_assistant:
            msgs = msgs + [{"role": "assistant", "content": ""}]
        audios, _, _ = process_mm_info(msgs, use_audio_in_video=True)

        prompt_only_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        if self.split == "train":
            full_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = self.tokenizer(prompt_only_text, audio=audios, add_special_tokens=False)["input_ids"][0]
            out = self.tokenizer(full_text, audio=audios, add_special_tokens=False)
            input_ids = out["input_ids"][0]
            attention_mask = out["attention_mask"][0]
            input_features = out["input_features"][0]
            feature_attention_mask = out["feature_attention_mask"][0]
            # We align to current input_ids length, but prompt_ids might be longer than max_len.
            p_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * len(input_ids)
            for i in range(p_len, len(input_ids)):
                labels[i] = input_ids[i]
        else:
            prompt_ids = self.tokenizer(prompt_only_text, audio=audios, add_special_tokens=False)
            input_ids = prompt_ids["input_ids"][0]
            attention_mask = prompt_ids["attention_mask"][0]
            input_features = prompt_ids["input_features"][0]
            feature_attention_mask = prompt_ids["feature_attention_mask"][0]
            labels = None
        ref_answer = msgs[-1]["content"][0]["text"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_features": torch.tensor(input_features, dtype=torch.bfloat16),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
            "feature_attention_mask": torch.tensor(feature_attention_mask, dtype=torch.long),
            "ref_answer": ref_answer
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        datapiece = self._build_prompt_and_full_text(sample, self.tokenizer)
        return datapiece

    def __len__(self) -> int:
        return len(self.data)


# -----------------------------
# Collator (pad to multiples of 8 for TensorCores)
# -----------------------------
class PadToMultipleCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        # Convert tensors -> lists for tokenizer.pad
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "input_features": [],
            "feature_attention_mask": [],
            "ref_answer": []
        }
        for f in features:
            batch["input_ids"].append(f["input_ids"])
            batch["attention_mask"].append(f["attention_mask"])
            batch["labels"].append(f["labels"])
            batch["input_features"].append(f["input_features"])
            batch["feature_attention_mask"].append(f["feature_attention_mask"])
            batch["ref_answer"].append(f["ref_answer"])

        batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_value=0)
        batch["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=0)
        batch["labels"] = pad_sequence(batch["labels"], batch_first=True, padding_value=-100) if batch["labels"][0] is not None else None
        batch["input_features"] = pad_sequence(batch["input_features"], batch_first=True, padding_value=0)
        batch["feature_attention_mask"] = pad_sequence(batch["feature_attention_mask"], batch_first=True, padding_value=0)
        return batch