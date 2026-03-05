#!/usr/bin/env python3

import sys, os
sys.path.append('third_party/Matcha-TTS')
sys.path.append('./')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import torchaudio
from rich import progress_bar
import json
import numpy as np
import torchaudio.transforms as T
from pathlib import Path
import argparse


# Custom argument type to handle flexible boolean inputs
def str2bool(v):
    """Convert various string representations to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def test(cosyvoice, ref_audio, prompt, target, gen_audio_save_path):
    prompt_speech_16k = load_wav(ref_audio, 16000)
    for i, j in enumerate(cosyvoice.inference_instruct2(target, prompt, prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save(gen_audio_save_path, j['tts_speech'], cosyvoice.sample_rate)


def test_cfg(cosyvoice, ref_audio, prompt, target, gen_audio_save_path, drop_prompt, drop_target, cfg_scale, reference_prompt=None):
    prompt_speech_16k = load_wav(ref_audio, 16000)
    if not reference_prompt:
        for i, j in enumerate(cosyvoice.inference_instruct3(target, prompt, prompt_speech_16k, stream=False, text_frontend=False, cfg_scale=cfg_scale, drop_prompt=drop_prompt, drop_target=drop_target)):
            torchaudio.save(gen_audio_save_path, j['tts_speech'], cosyvoice.sample_rate)
    else:
        for i, j in enumerate(cosyvoice.inference_instruct4(target, prompt, reference_prompt, prompt_speech_16k, stream=False, text_frontend=False, cfg_scale=cfg_scale, drop_prompt=drop_prompt, drop_target=drop_target)):
            torchaudio.save(gen_audio_save_path, j['tts_speech'], cosyvoice.sample_rate)

def test_cfg_filter(cosyvoice, ref_audio, prompt, target, gen_audio_save_path, drop_prompt, drop_target, cfg_scale, top_k, cfg_rescale=1.0, reference_prompt=None):
    prompt_speech_16k = load_wav(ref_audio, 16000)
    for i, j in enumerate(cosyvoice.inference_instruct3(target, prompt, prompt_speech_16k, stream=False, text_frontend=False, cfg_scale=cfg_scale, drop_prompt=drop_prompt, drop_target=drop_target, filter_topk=top_k, cfg_rescale=cfg_rescale)):
        torchaudio.save(gen_audio_save_path, j['tts_speech'], cosyvoice.sample_rate)
    else:
        for i, j in enumerate(cosyvoice.inference_instruct4(target, prompt, reference_prompt, prompt_speech_16k, stream=False, text_frontend=False, cfg_scale=cfg_scale, drop_prompt=drop_prompt, drop_target=drop_target)):
            torchaudio.save(gen_audio_save_path, j['tts_speech'], cosyvoice.sample_rate)


def form_prompt(emotion):
    if emotion == "neutral":
        return f"Speaks in a neutral tone."
    else:
        return f"Speaks in a strongly {emotion} tone."

def main():
    # json_path = "./data/random_test.jsonl"
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Input Jsonl file that contains cfg calculated using NLI model")
    parser.add_argument("--model_path", default="pretrained_models/CosyVoice2-0.5B", help="Path to the CosyVoice2 model directory")
    parser.add_argument("--save_path", default="data/gen_instruct", help="Path to save generated audio files")
    parser.add_argument("--cfg_scale", default=1.0, type=float, help="CFG scale for generation")
    parser.add_argument("--target_emotion", type=str, default=None, help="Utt to gpt predicted text-based emotion")
    parser.add_argument("--drop_prompt", type=str2bool, default=False, help="Whether to drop the prompt during generation. Accepts true/false, yes/no, 1/0")
    parser.add_argument("--drop_target", type=str2bool, default=False, help="Whether to drop the target during generation. Accepts true/false, yes/no, 1/0")
    parser.add_argument("--filter_topk", type=int, default=-1, help="Top-k filtering for CFG. Use -1 to disable filtering.")
    parser.add_argument("--rescale_cfg", type=float, default=1.0, help="Rescale factor for CFG.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible TTS generation. Use --seed 42 for reproducibility.")
    
    args = parser.parse_args()
    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)
    cosyvoice = CosyVoice2(args.model_path, load_jit=False, load_trt=False, fp16=False)
    json_path = args.json_path
    utt2target_emo = None
    if args.target_emotion is not None:
        utt2target_emo = {}
        with open(args.target_emotion, 'r') as f:
            for line in f:
                utt, target_emo = line.strip().split("\t")
                utt2target_emo[utt] = target_emo
    if args.seed is not None:
        set_all_random_seed(args.seed)
    with open(json_path, 'r') as f:
        json_lines = f.readlines()
        # {"ID": "libritts-train-clean-360-3009-10328-3009_10328_000023_000002", "audio_path": "mnt/data/yizhou/datasets/libritts/train-clean-360/3009/10328/3009_10328_000023_000002.wav", "gender": "M", "pitch": "low", "energy": "normal", "emotion": "neutral", "style_prompt": "A male speaker employs a deep voice to converse naturally with an average speaking tempo and normal vigor.", "content_prompt": "Yet Divine Scripture from time to time introduces angels so apparent as to be seen commonly by all; just as the angels who appeared to Abraham were seen by him and by his whole family, by Lot, and by the citizens of Sodom; in like manner the angel who appeared to Tobias was seen by all present.", "file_exists": true}
        for line in json_lines:
            try:
                json_data = json.loads(line)
                
                ref_audio = Path(json_data['ref_audio_path'])
                ref_text = json_data['ref_content']
                ref_emotion = json_data['ref_emotion']
                target_text = json_data['content']
                target_emotion = json_data['emotion']
                gpt_pred_emo = utt2target_emo.get(json_data['ID'], "") if utt2target_emo is not None else ""
                same_emotion = gpt_pred_emo == target_emotion if gpt_pred_emo != "" else False
                
                target_prompt = form_prompt(target_emotion)
                if gpt_pred_emo != "" and not same_emotion:
                    reference_prompt = form_prompt(gpt_pred_emo)
                else:
                    reference_prompt = None
                # prompt = json_data['style_prompt']
                # target = json_data['content_prompt']
                print(reference_prompt, ref_audio, target_prompt, target_text)
                gen_audio_save_path = os.path.join(args.save_path, f"{json_data['ID']}.wav")
                if reference_prompt is None:
                    if args.cfg_scale == 1.0:
                        test(cosyvoice, ref_audio, target_prompt, target_text, gen_audio_save_path)
                    elif args.filter_topk == -1:
                        test_cfg(cosyvoice, ref_audio, target_prompt, target_text, gen_audio_save_path, drop_prompt=args.drop_prompt, drop_target=args.drop_target, cfg_scale=args.cfg_scale)
                    else:
                        test_cfg_filter(cosyvoice, ref_audio, target_prompt, target_text, gen_audio_save_path, drop_prompt=args.drop_prompt, drop_target=args.drop_target, cfg_scale=args.cfg_scale, top_k=args.filter_topk, cfg_rescale=args.rescale_cfg)
                else:
                    if args.cfg_scale == 1.0:
                        test(cosyvoice, ref_audio, target_prompt, target_text, gen_audio_save_path)
                    elif args.filter_topk == -1:
                        test_cfg(cosyvoice, ref_audio, target_prompt, target_text, gen_audio_save_path, drop_prompt=args.drop_prompt, drop_target=args.drop_target, cfg_scale=args.cfg_scale, reference_prompt=reference_prompt)
                    else:
                        test_cfg_filter(cosyvoice, ref_audio, target_prompt, target_text, gen_audio_save_path, drop_prompt=args.drop_prompt, drop_target=args.drop_target, cfg_scale=args.cfg_scale, top_k=args.filter_topk, cfg_rescale=args.rescale_cfg, reference_prompt=reference_prompt)
            except Exception as e:
                sample_id = "unknown"
                try:
                    sample_id = json.loads(line).get("ID", "unknown")
                except Exception:
                    pass
                print(f"ERROR: Failed to generate sample {sample_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    
    main()
