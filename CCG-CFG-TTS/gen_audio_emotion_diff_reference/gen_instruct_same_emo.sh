#!/usr/bin/env bash
#SBATCH --job-name=gen_instruct_same_emo
#SBATCH -o ./gen_audio_emotion_diff_reference/log/gen_instruct_same_emo.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# --- Configurable paths (set via environment or edit defaults below) ---
EMOTION2VEC_DIR="${EMOTION2VEC_DIR:-tools/emotion2vec}"
UTMOS_DIR="${UTMOS_DIR:-tools/UTMOS}"
UTMOS_CKPT="${UTMOS_CKPT:-${UTMOS_DIR}/epoch=3-step=7459.ckpt}"
NISQA_DIR="${NISQA_DIR:-tools/NISQA}"
DATA_DIR="${DATA_DIR:-gen_audio_emotion_diff_reference/data}"
TEST_JSONL="${DATA_DIR}/test_jsonl/test.same-emo.jsonl"
KALDI_TEST_DIR="${DATA_DIR}/test_jsonl/test"
MODEL_PATH="${MODEL_PATH:-pretrained_models/CosyVoice2-0.5B}"
# -----------------------------------------------------------------------

mkdir -p gen_audio_emotion_diff_reference/data gen_audio_emotion_diff_reference/log
output_audio=gen_audio_emotion_diff_reference/data/gen_instruct_same_emo
output_dir=gen_audio_emotion_diff_reference/data/gen_instruct_same_emo_metadata
output_dir_test=$output_dir/test
export LOCAL_RANK=0

# Step 1: Generate audios (no CFG, cfg_scale=1.0)
CUDA_VISIBLE_DEVICES="0" python gen_audio_emotion_diff_reference/gen_dataset.py \
  --json_path "$TEST_JSONL" --model_path "$MODEL_PATH" --save_path $output_audio \
  --cfg_scale 1.0 --drop_prompt 0 --drop_target 0 --filter_topk -1 --rescale_cfg 1.0

# Step 2: Kaldi format conversion
mkdir -p $output_dir
cp -r "$KALDI_TEST_DIR" $output_dir
find $PWD/$output_audio -type f -name "*.wav" > $output_dir_test/wavlist
cat $output_dir_test/wavlist | rev | cut -d "/" -f1 | rev | sed 's/\.wav//g' | paste -d " " - $output_dir_test/wavlist > $output_dir_test/wav.scp
cut -d " " -f1 $output_dir_test/utt2emotion > $output_dir_test/utt.list
grep -F -f $output_dir_test/utt.list $output_dir_test/wav.scp > $output_dir_test/wav.scp.filtered
mv $output_dir_test/wav.scp.filtered $output_dir_test/wav.scp

# Step 3: emotion2vec accuracy
CUDA_VISIBLE_DEVICES="0" python "$EMOTION2VEC_DIR/test_dataset.py" --wav_scp $output_dir_test/wav.scp --output_dir $output_dir_test
python "$EMOTION2VEC_DIR/calculate_accuracy.py" --results $output_dir_test/emo_results.txt --ground_truth $output_dir_test/utt2emotion --output $output_dir_test/emo.acc

# Step 4: UTMOS
CUDA_VISIBLE_DEVICES="0" python "$UTMOS_DIR/predict.py" --ckpt_path "$UTMOS_CKPT" --mode predict_dir --inp_dir $output_audio --bs 200 --out_path $output_dir_test/utmos.tsv

# Step 5: Whisper WER
CUDA_VISIBLE_DEVICES="0" python ./scripts/get_wer.py --wav_scp $output_dir_test/wav.scp --output_result $output_dir_test/recog.result.whisperlv3 --reference $output_dir_test/text

# Step 6: NISQA
CUDA_VISIBLE_DEVICES="0" python "$NISQA_DIR/run_predict.py" --mode predict_wavscp --pretrained_model "$NISQA_DIR/weights/nisqa.tar" --wav_scp $output_dir_test/wav.scp --num_workers 2 --bs 100 --output_dir $output_dir_test
CUDA_VISIBLE_DEVICES="0" python "$NISQA_DIR/run_predict.py" --mode predict_wavscp --pretrained_model "$NISQA_DIR/weights/nisqa_tts.tar" --wav_scp $output_dir_test/wav.scp --num_workers 2 --bs 100 --output_dir $output_dir_test

# Step 7: DNSMOS
CUDA_VISIBLE_DEVICES="0" python ./scripts/compute_dnsmos.py --mode wavscp --wav_scp $output_dir_test/wav.scp -o $output_dir_test --sample-rate 24000 --device "cuda:0"
