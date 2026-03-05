#!/usr/bin/env python

import argparse
import pathlib
import tqdm
import json
import os
import math
from collections import defaultdict

def estimate_syllables(text: str) -> int:
    """
    Simple English syllable estimator based on vowel groups.
    """
    text = text.lower()
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in text:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    # Heuristic adjustments
    if text.endswith("e") and count > 1:
        count -= 1
    if count == 0:
        count = 1
    return count

def get_bin_for_syllable_count(syllable_count: int, count_stats: dict) -> dict:
    """
    Find the bin that contains the given syllable count.
    Returns the bin stats dict or None if not found.
    """
    bins = count_stats.get("bins", [])
    for idx, bin_data in enumerate(bins):
        bin_range = bin_data["bin_range"]
        # Parse range like "0-10", "10-20", etc.
        if "-" in bin_range:
            lower, upper = bin_range.split("-")
            lower = int(lower)
            upper = int(upper)
            # Last bin is inclusive on both ends, others are [lower, upper)
            if idx == len(bins) - 1:
                if lower <= syllable_count <= upper:
                    return bin_data
            else:
                if lower <= syllable_count < upper:
                    return bin_data
    return None

def form_prompt(emotion: str) -> str:
    """Format style prompt from emotion label."""
    if emotion is None:
        return "Speaks in a neutral tone."
    emotion = str(emotion).lower()
    if emotion == "neutral":
        return "Speaks in a neutral tone."
    else:
        return f"Speaks in a strongly {emotion} tone."

def calculate_syllable_dur_reward(actual_dur_per_syll: float, mean: float, std: float) -> float:
    """
    Calculate reward using formula: z = |x - μ| / σ, reward = exp(-z)
    """
    if std == 0 or std is None or mean is None:
        return 1.0 if actual_dur_per_syll == mean else 0.0
    
    z = abs(actual_dur_per_syll - mean) / std
    reward = math.exp(-z)
    return reward

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_gt_dir", default="data/train_shorter_than_30s", type=pathlib.Path)
    parser.add_argument("--inp_gen_dirs", nargs='+', help="List of generated audio directories. Can be named, e.g., 'gen1:path/to/gen1 gen2:path/to/gen2'", required=True)
    parser.add_argument("--style_prompt", default="./data/train_generated_audios/utt2style", type=pathlib.Path)
    parser.add_argument("--out_dir", default="data/dpo_samples_v3", type=pathlib.Path)
    parser.add_argument("--utt_match", type=pathlib.Path, default=None,
                        help="Optional uttlist file: one uttid per line, where target emotion and text emotion matches. "
                             "For these uttids, pos samples are restricted to inp_gt_dir and inp_gen_dirs without 'cfg' in path.")
    
    # Reference JSON files for syllable duration constraints
    parser.add_argument("--emo_syllable_dur_stats", 
                       default="data/train_shorter_than_30s/emotion_syllable_duration_stats.json",
                       type=pathlib.Path,
                       help="Path to emotion syllable duration statistics JSON file")
    parser.add_argument("--count_syllable_dur_stats",
                       default="data/train_shorter_than_30s/syllable_duration_by_count.json",
                       type=pathlib.Path,
                       help="Path to syllable duration by count statistics JSON file")

    # Original weights
    parser.add_argument("--emo_weight", type=float, default=0.4)
    parser.add_argument("--utmos_weight", type=float, default=0.2)
    parser.add_argument("--wer_weight", type=float, default=0.4)
    parser.add_argument("--speed_warp_weight", type=float, default=-0.2,
                        help="Negative weight for speed warp score.")
    
    # NEW: weights for syllable duration constraints
    parser.add_argument("--emo_syllable_dur_weight", type=float, default=0.1,
                        help="Weight for emotion-based syllable duration reward.")
    parser.add_argument("--count_syllable_dur_weight", type=float, default=0.1,
                        help="Weight for count-based syllable duration reward.")

    # NEW: enable/disable flags per factor (default: enabled)
    parser.add_argument("--disable_emo", action="store_true",
                        help="Disable emotion certainty factor in final score.")
    parser.add_argument("--disable_utmos", action="store_true",
                        help="Disable UTMOS factor in final score.")
    parser.add_argument("--disable_wer", action="store_true",
                        help="Disable WER factor in final score.")
    parser.add_argument("--disable_speed_warp", action="store_true",
                        help="Disable speed warp factor in final score.")
    parser.add_argument("--disable_emo_syllable_dur", action="store_true",
                        help="Disable emotion-based syllable duration factor.")
    parser.add_argument("--disable_count_syllable_dur", action="store_true",
                        help="Disable count-based syllable duration factor.")

    # NEW: hard thresholds for each factor
    parser.add_argument("--emo_min_certainty", type=float, default=None,
                        help="If set, drop samples whose emotion certainty < this value "
                             "(only checked when emotion factor is enabled).")
    parser.add_argument("--utmos_min", type=float, default=None,
                        help="If set, drop samples whose UTMOS score < this value "
                             "(only checked when UTMOS factor is enabled).")
    parser.add_argument("--wers_min", type=float, default=None,
                        help="If set, positive samples must have WER-based score >= this value "
                             "(negative samples are NOT excluded by this; only checked when WER factor is enabled).")
    parser.add_argument("--speed_warp_max", type=float, default=None,
                        help="If set, drop samples whose speed warp score > this value "
                             "(only checked when speed-warp factor is enabled).")

    parser.add_argument("--no_neg_strict_constraint", action="store_true",
                        help="If set, disable the constraint that negative must have both wers and emo_certainty worse than positive.")

    return parser.parse_args()

def load_reward_scores(data_path: pathlib.Path):
    reward_scores = {}
    
    # if "Cosyvoice2" in str(data_path):
    data_path = data_path / "test"
    emotion_certainty_path = data_path / "emo_results.txt"
    utmos_path = data_path / "utmos.tsv"
    wer_path = data_path / "recog.result.whisperlv3.wer"
    wav_scp = data_path / "wav.scp"
    text = data_path / "text"
    speed_warp_path = data_path / "utt2spwrp"
    utt2dur_path = data_path / "utt2dur"
    utt2emotion_path = data_path / "utt2emotion"
    
    if not data_path.exists():
        print(f"Warning: Data path does not exist: {data_path}")
        return None

    emo_scores = {}
    if emotion_certainty_path.exists():
        with open(emotion_certainty_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                uttid, emotion, certainty_score = parts[0], parts[1], float(parts[2])
                emo_scores[uttid] = (emotion, certainty_score)
    reward_scores["emo"] = emo_scores
            
    utmos = {}
    if utmos_path.exists():
        with open(utmos_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                uttid, utmos_score = parts[0], float(parts[1])
                utmos[uttid] = utmos_score / 5.0
    reward_scores["utmos"] = utmos
            
    wers_score = {}
    if wer_path.exists():
        with open(wer_path, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 6):
                try:
                    utt_line = lines[i+1].strip()
                    wer_line = lines[i+2].strip()
                    uttid = utt_line.split(":", 1)[1].strip()
                    wer_percent = float(wer_line.split()[1])
                    wers_score[uttid] = 1.0 - (wer_percent / 100.0)
                except (IndexError, ValueError) as e:
                    continue
    reward_scores["wers"] = wers_score

    speed_warp_scores = {}
    if speed_warp_path.exists():
        with open(speed_warp_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                uttid, score = parts[0], float(parts[1])
                speed_warp_scores[uttid] = score
    reward_scores["speed_warp"] = speed_warp_scores
    
    wav_paths = {}
    if wav_scp.exists():
        with open(wav_scp, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                wav_paths[parts[0]] = parts[1]
    reward_scores["wavs"] = wav_paths

    content_prompts = {}
    if text.exists():
        with open(text, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2: continue
                content_prompts[parts[0]] = parts[1]
    reward_scores["content"] = content_prompts

    # Load utt2dur for syllable duration calculation
    utt2dur = {}
    if utt2dur_path.exists():
        with open(utt2dur_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                utt2dur[parts[0]] = float(parts[1])
    reward_scores["utt2dur"] = utt2dur

    # Load utt2emotion for emotion-based syllable duration
    utt2emotion = {}
    if utt2emotion_path.exists():
        with open(utt2emotion_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                utt2emotion[parts[0]] = parts[1]
    reward_scores["utt2emotion"] = utt2emotion

    return reward_scores

def main():
    args = get_arg()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference statistics
    print(f"Loading emotion syllable duration stats from {args.emo_syllable_dur_stats}...")
    with open(args.emo_syllable_dur_stats, "r") as f:
        emo_syllable_dur_stats = json.load(f)
    print(f"Loaded stats for {len(emo_syllable_dur_stats)} emotions")
    
    print(f"Loading count-based syllable duration stats from {args.count_syllable_dur_stats}...")
    with open(args.count_syllable_dur_stats, "r") as f:
        count_syllable_dur_stats = json.load(f)
    print(f"Loaded stats for {len(count_syllable_dur_stats.get('bins', []))} bins")
    
    gen_dirs = {}
    for d in args.inp_gen_dirs:
        if ":" in d:
            name, path = d.split(":", 1)
            gen_dirs[name] = pathlib.Path(path)
        else:
            path = pathlib.Path(d)
            gen_dirs[path.name] = path

    # Load utt_match list (uttids where target emotion and text emotion matches)
    utt_match_set = set()
    if args.utt_match is not None and args.utt_match.exists():
        with open(args.utt_match, "r") as f:
            for line in f:
                uttid = line.strip().split()[0]
                if uttid:
                    utt_match_set.add(uttid)
        print(f"Loaded {len(utt_match_set)} uttids from utt_match list")

    # For utt_match uttids: pos samples allowed only from inp_gt_dir and gen_dirs without "cfg" in path
    utt_match_pos_sources = {"gt"}
    for name, path in gen_dirs.items():
        if "cfg" not in str(path):
            utt_match_pos_sources.add(name)
    # ipdb.set_trace()
    all_scores = {"gt": load_reward_scores(args.inp_gt_dir)}
    for name, path in gen_dirs.items():
        scores = load_reward_scores(path)
        if scores:
            all_scores[name] = scores
    
    pos_file = args.out_dir / "positive_samples.jsonl"
    neg_file = args.out_dir / "negative_samples.jsonl"
    skipped_file = args.out_dir / "skipped_samples.jsonl"
    emo_labels = {}
    emotion_labels_path = args.inp_gt_dir / "test" / "utt2emotion"
    if emotion_labels_path.exists():
        with open(emotion_labels_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                emo_labels[parts[0]] = parts[1]
    total_pos = 0
    total_neg = 0
    pos_counts = defaultdict(int)  # per-source positive counts
    neg_counts = defaultdict(int)  # per-source negative counts
    processed_utt = 0
    skipped_utt = 0
    
    with open(pos_file, "w") as pos_fh, open(neg_file, "w") as neg_fh, open(skipped_file, "w") as skipped_fh:
        gt_uttids = set(all_scores["gt"]["emo"].keys())
        for uttid in tqdm.tqdm(gt_uttids):
            if uttid not in emo_labels:
                skipped_utt += 1
                skipped_fh.write(json.dumps({"id": uttid, "reason": "missing_emotion_label"}) + "\n")
                continue

            samples = []

            # Process all available samples (gt and generated)
            for name, scores in all_scores.items():
                if not scores or uttid not in scores.get("wavs", {}) or uttid not in scores.get("content", {}):
                    continue

                emo_label, emo_certainty = scores.get("emo", {}).get(uttid, (None, 0.0))

                # Mark whether this sample's recognized emotion matches ground-truth
                emotion_correct = (emo_label == emo_labels[uttid]) if emo_label is not None else False
                # If emotion is wrong, emo_certainty is set to 0 (no credit for confidently wrong)
                if not emotion_correct:
                    emo_certainty = 0.0

                utmos_score = scores.get("utmos", {}).get(uttid, 0.0)
                wers_score = scores.get("wers", {}).get(uttid, 0.0)
                speed_warp_score = scores.get("speed_warp", {}).get(uttid, 0.0)

                # Calculate syllable duration rewards
                emo_syllable_dur_reward = 0.0
                count_syllable_dur_reward = 0.0
                
                # Get text and duration for syllable calculations
                text = scores.get("content", {}).get(uttid, "")
                duration = scores.get("utt2dur", {}).get(uttid, 0.0)
                emotion = scores.get("utt2emotion", {}).get(uttid, None)
                
                if text and duration > 0:
                    syllable_count = estimate_syllables(text)
                    if syllable_count > 0:
                        dur_per_syll = duration / syllable_count
                        
                        # Calculate emotion-based syllable duration reward
                        if not args.disable_emo_syllable_dur and emotion:
                            emotion_lower = emotion.lower()
                            if emotion_lower in emo_syllable_dur_stats:
                                emo_stats = emo_syllable_dur_stats[emotion_lower]
                                mean = emo_stats["mean_syllable_duration_sec"]
                                std = emo_stats["std_syllable_duration_sec"]
                                emo_syllable_dur_reward = calculate_syllable_dur_reward(dur_per_syll, mean, std)
                        
                        # Calculate count-based syllable duration reward
                        if not args.disable_count_syllable_dur:
                            bin_data = get_bin_for_syllable_count(syllable_count, count_syllable_dur_stats)
                            if bin_data and bin_data.get("mean_syllable_duration_sec") is not None:
                                mean = bin_data["mean_syllable_duration_sec"]
                                std = bin_data["std_syllable_duration_sec"]
                                count_syllable_dur_reward = calculate_syllable_dur_reward(dur_per_syll, mean, std)

                # ---- HARD THRESHOLDS (drop sample if it fails any active constraint) ----
                if (not args.disable_emo) and (args.emo_min_certainty is not None):
                    if emo_certainty < args.emo_min_certainty:
                        continue

                if (not args.disable_utmos) and (args.utmos_min is not None):
                    if utmos_score < args.utmos_min:
                        continue

                # wers_min applies only to positive selection; negative can use bad-WER samples

                if (not args.disable_speed_warp) and (args.speed_warp_max is not None):
                    if speed_warp_score > args.speed_warp_max:
                        continue

                # ---- FINAL SCORE WITH PER-FACTOR SWITCHES ----
                final_score = 0.0
                if not args.disable_emo:
                    final_score += emo_certainty * args.emo_weight
                if not args.disable_utmos:
                    final_score += utmos_score * args.utmos_weight
                if not args.disable_wer:
                    final_score += wers_score * args.wer_weight
                if not args.disable_speed_warp:
                    final_score += speed_warp_score * args.speed_warp_weight
                if not args.disable_emo_syllable_dur:
                    final_score += emo_syllable_dur_reward * args.emo_syllable_dur_weight
                if not args.disable_count_syllable_dur:
                    final_score += count_syllable_dur_reward * args.count_syllable_dur_weight

                sample_data = {
                    "ID": uttid,
                    "audio_path": scores["wavs"][uttid],
                    "style_prompt": form_prompt(emo_labels.get(uttid)),
                    "content_prompt": scores["content"][uttid],
                    "score": final_score,
                    "source": name,
                    "emotion_correct": emotion_correct,
                    "wers_score": wers_score,
                    "emo_certainty": emo_certainty,
                    "emo_syllable_dur_reward": emo_syllable_dur_reward,
                    "count_syllable_dur_reward": count_syllable_dur_reward
                }
                samples.append(sample_data)

            if len(samples) < 2:
                skipped_utt += 1
                skipped_fh.write(json.dumps({"id": uttid, "reason": "insufficient_samples"}) + "\n")
                continue

            # Sort samples by score in descending order for selection
            samples.sort(key=lambda x: x["score"], reverse=True)

            correct_samples = [s for s in samples if s.get("emotion_correct", False)]
            incorrect_samples = [s for s in samples if not s.get("emotion_correct", False)]

            # For utt_match uttids: restrict pos candidates to inp_gt_dir and gen_dirs without "cfg" in path
            if uttid in utt_match_set:
                correct_samples = [s for s in correct_samples if s["source"] in utt_match_pos_sources]
                pos_candidate_samples = [s for s in samples if s["source"] in utt_match_pos_sources]
            else:
                pos_candidate_samples = samples  # no restriction

            # wers_min applies only to positive: filter pos candidates to those meeting WER requirement
            if (not args.disable_wer) and (args.wers_min is not None):
                correct_samples = [s for s in correct_samples if s.get("wers_score", 0.0) >= args.wers_min]
                pos_candidate_samples = [s for s in pos_candidate_samples if s.get("wers_score", 0.0) >= args.wers_min]
            # negative samples are NOT filtered by wers_min

            # Three-case handling generalized to multi-source:
            # 1) No accurate emotions (or no valid pos candidates) -> fallback: gt as positive, lowest score as negative
            if len(correct_samples) == 0:
                gt_samples = [s for s in samples if s["source"] == "gt"]
                if len(gt_samples) == 0:
                    skipped_utt += 1
                    skipped_fh.write(json.dumps({"id": uttid, "reason": "no_emotion_correct_samples_no_gt"}) + "\n")
                    continue
                positive_sample = gt_samples[0]
                neg_candidates = [s for s in samples if s is not positive_sample]
                if len(neg_candidates) == 0:
                    skipped_utt += 1
                    skipped_fh.write(json.dumps({"id": uttid, "reason": "no_emotion_correct_samples_no_neg"}) + "\n")
                    continue
                negative_sample = min(neg_candidates, key=lambda x: x["score"])  # lowest score
                pos_fh.write(json.dumps(positive_sample) + '\n')
                neg_fh.write(json.dumps(negative_sample) + '\n')
                total_pos += 1
                total_neg += 1
                pos_counts[positive_sample["source"]] += 1
                neg_counts[negative_sample["source"]] += 1
                processed_utt += 1
                continue
            # 2) Mix of accurate and inaccurate -> choose best accurate as positive, best inaccurate as negative
            if len(incorrect_samples) > 0:
                positive_sample = correct_samples[0]  # already sorted desc by score
                neg_candidates = incorrect_samples
            else:
                # 3) All accurate -> choose best as positive, worst as negative
                if uttid in utt_match_set and len(pos_candidate_samples) > 0:
                    positive_sample = pos_candidate_samples[0]
                else:
                    positive_sample = samples[0]
                neg_candidates = samples

            # Constraint: negative must have BOTH wers_score AND emo_certainty worse (strictly lower) than positive
            neg_candidates = [s for s in neg_candidates if s is not positive_sample]
            neg_candidates_sorted = sorted(neg_candidates, key=lambda x: x["score"], reverse=True)
            if not args.no_neg_strict_constraint:
                pos_wers = positive_sample.get("wers_score", 0.0)
                valid_neg = [s for s in neg_candidates
                             if s.get("wers_score", 0.0) < pos_wers]
                if len(valid_neg) > 0:
                    neg_candidates = valid_neg
                    neg_candidates_sorted = sorted(neg_candidates, key=lambda x: x["score"], reverse=True)
            # Pick worst (lowest score) among candidates
            negative_sample = neg_candidates_sorted[-1]

            pos_fh.write(json.dumps(positive_sample) + '\n')
            neg_fh.write(json.dumps(negative_sample) + '\n')
            total_pos += 1
            total_neg += 1
            pos_counts[positive_sample["source"]] += 1
            neg_counts[negative_sample["source"]] += 1
            processed_utt += 1

    print(f"Positive samples written to {pos_file}")
    print(f"Negative samples written to {neg_file}")
    print(f"Skipped samples (id, reason) written to {skipped_file}")
    print(f"Total utterances processed: {processed_utt}")
    print(f"Total utterances skipped (missing labels or insufficient candidates): {skipped_utt}")
    print(f"Total positive samples: {total_pos}")
    print(f"Total negative samples: {total_neg}")

    # Per-source breakdown (gt and each generated set)
    print("Positive samples by source:")
    for name in all_scores.keys():
        print(f"  {name}: {pos_counts.get(name, 0)}")

    print("Negative samples by source:")
    for name in all_scores.keys():
        print(f"  {name}: {neg_counts.get(name, 0)}")
    
if __name__ == "__main__":
    main()

