#!/usr/bin/env python3
"""
Calculate emotion recognition accuracy by comparing predictions with ground truth labels.
Compares emo_results.txt (from test_dataset.py) with utt2emotion ground truth.
"""

import argparse
import os
import json
from collections import defaultdict, Counter


def load_ground_truth(utt2emotion_file):
    ground_truth = {}
    with open(utt2emotion_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    utt_id, emotion = parts
                    ground_truth[utt_id] = emotion
    return ground_truth


def load_predictions(results_file):
    predictions = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                if len(parts) >= 3:
                    utt_id = parts[0]
                    predicted_emotion = parts[1]
                    confidence = float(parts[2])
                    predictions[utt_id] = (predicted_emotion, confidence)
    return predictions


def calculate_accuracy(ground_truth, predictions):
    common_utts = set(ground_truth.keys()) & set(predictions.keys())

    if not common_utts:
        print("Warning: No common utterances found between ground truth and predictions!")
        return {}

    correct = 0
    total = len(common_utts)
    emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predicted': 0})
    confusion_data = defaultdict(lambda: defaultdict(int))

    for utt_id in common_utts:
        true_emotion = ground_truth[utt_id]
        pred_emotion, confidence = predictions[utt_id]

        if true_emotion == pred_emotion:
            correct += 1
            emotion_stats[true_emotion]['correct'] += 1

        emotion_stats[true_emotion]['total'] += 1
        emotion_stats[pred_emotion]['predicted'] += 1
        confusion_data[true_emotion][pred_emotion] += 1

    overall_accuracy = correct / total

    per_emotion_metrics = {}
    for emotion in emotion_stats:
        stats = emotion_stats[emotion]
        precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0.0
        recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_emotion_metrics[emotion] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': stats['total']
        }

    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'per_emotion_metrics': per_emotion_metrics,
        'confusion_matrix': dict(confusion_data),
        'common_utterances': len(common_utts),
        'gt_utterances': len(ground_truth),
        'pred_utterances': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate emotion recognition accuracy')
    parser.add_argument('--results', required=True, help='Path to emo_results.txt')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth utt2emotion file')
    parser.add_argument('--output', help='Path to save detailed results as JSON (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return

    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        return

    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.results)
    metrics = calculate_accuracy(ground_truth, predictions)

    if metrics:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
