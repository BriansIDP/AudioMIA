import soundfile as sf
import argparse
import sys, os
import json
from tqdm import tqdm
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
from mia_utils import get_meta_metrics, get_img_metric



# Custom stopping criteria
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        last_tokens = input_ids[:, -1]
        return (last_tokens == self.stop_token_id).all()

stop_token_id = 151645
stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "t"}

def prediction_step(args, model, processor, audiopath, prompt, return_dict_in_generate=False):
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
                {"type": "text", "text": prompt}
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    return_audio=False,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    logits = None
    if args.from_audio:
        with torch.no_grad():
            outputs = model(**inputs)
            audio_positions = torch.where(inputs["input_ids"] == 151646)[1]
            logits = outputs.logits[0, audio_positions]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
            sudo_input_ids = logits.argmax(dim=-1)
            # Forward reversed features
            feature_end = inputs["feature_attention_mask"].sum()
            reverse_feature = inputs["input_features"][:, :, :feature_end]
            reverse_feature = torch.flip(reverse_feature, dims=[2])
            reverse_feature = torch.cat([reverse_feature, inputs["input_features"][:, :, feature_end:]], dim=-1)
            inputs["input_features"] = reverse_feature
            reverse_outputs = model(**inputs)
            reverse_logits = reverse_outputs.logits[0, audio_positions]
            reverse_probabilities = torch.nn.functional.softmax(reverse_logits, dim=-1)
            reverse_log_probabilities = torch.nn.functional.log_softmax(reverse_logits, dim=-1)
            revrese_sudo_input_ids = reverse_logits.argmax(dim=-1)

        metrics_dict = get_meta_metrics(sudo_input_ids, probabilities, log_probabilities)
        metrics_reverse = get_meta_metrics(revrese_sudo_input_ids, reverse_probabilities, reverse_log_probabilities)
        pred = get_img_metric(
            metrics_dict["ppl"], metrics_dict["all_prob"], metrics_dict["loss"], metrics_dict["entropies"], 
            metrics_dict["modified_entropies"], metrics_dict["max_prob"], metrics_dict["probabilities"], 
            metrics_dict["gap_prob"], metrics_dict["renyi_05"], metrics_dict["renyi_2"], metrics_dict["log_probs"],
            metrics_dict["mod_renyi_05"], metrics_dict["mod_renyi_2"], metrics_dict["sequence_sm_entropy"], metrics_dict["sm_entropy"],
            metrics_reverse["sequence_sm_entropy"], metrics_reverse["sm_entropy"]
        )
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                stopping_criteria=stopping_criteria,
            )
            text_ids = outputs.sequences
            text_ids_generated = text_ids[0, inputs["input_ids"].shape[1]:]
            logits = torch.cat(outputs.scores, dim=0)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
            metrics_dict = get_meta_metrics(text_ids_generated, probabilities, log_probabilities)
            pred = get_img_metric(
                metrics_dict["ppl"], metrics_dict["all_prob"], metrics_dict["loss"], metrics_dict["entropies"], 
                metrics_dict["modified_entropies"], metrics_dict["max_prob"], metrics_dict["probabilities"], 
                metrics_dict["gap_prob"], metrics_dict["renyi_05"], metrics_dict["renyi_2"], metrics_dict["log_probs"],
                metrics_dict["mod_renyi_05"], metrics_dict["mod_renyi_2"], metrics_dict["sequence_sm_entropy"], metrics_dict["sm_entropy"],
                metrics_dict["sequence_sm_entropy"], metrics_dict["sm_entropy"]
            )
        text = processor.decode(text_ids_generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred = text[0]
    return pred

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--datapath", type=str, default="./dataset")
    args.add_argument("--output_dir", type=str, default="./exp")
    args.add_argument("--bare_question", type=str2bool, default=False)
    args.add_argument("--return_logits", type=str2bool, default=False)
    args.add_argument("--lora_r", type=int, default=32)
    args.add_argument("--lora_alpha", type=int, default=64)
    args.add_argument("--lora_dropout", type=float, default=0.05)
    args.add_argument("--lora_ckpt", type=str, default="no")
    args.add_argument("--from_audio", type=str2bool, default=False)
    args = args.parse_args()
    with open(args.datapath) as fin:
        data = json.load(fin)

    letters = ["A", "B", "C", "D", "E"]

    MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.lora_ckpt != "no":
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.to(torch.bfloat16)
        model = model.merge_and_unload()
        model.cuda()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    letter_ids = [processor(l).input_ids[0][0] for l in letters]

    for datapiece in tqdm(data):
        pred = prediction_step(
            args,
            model,
            processor,
            datapiece["audio"],
            datapiece["question"],
            return_dict_in_generate=args.return_logits,
        )
        datapiece["pred"] = pred
        if "answer" in datapiece:
            print("REF:", datapiece["answer"])
            print("PRED:", pred)
            print("="*89)
    tag = "no_generation" if args.from_audio else "generation"
    with open(os.path.join(args.output_dir, "mia_qwen25omni_{}.json".format(tag)), 'w') as fp:
        json.dump(data, fp, indent=4)