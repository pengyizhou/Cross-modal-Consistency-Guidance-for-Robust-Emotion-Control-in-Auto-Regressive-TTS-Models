#!/usr/bin/env bash

# --- Configurable paths (set via environment or edit defaults below) ---
EMOTION2VEC_DIR="${EMOTION2VEC_DIR:-tools/emotion2vec}"
UTMOS_DIR="${UTMOS_DIR:-tools/UTMOS}"
UTMOS_CKPT="${UTMOS_CKPT:-${UTMOS_DIR}/epoch=3-step=7459.ckpt}"
NISQA_DIR="${NISQA_DIR:-tools/NISQA}"
DATA_DIR="${DATA_DIR:-gen_audio_emotion_diff_reference/data}"
TEST_JSONL="${DATA_DIR}/test_jsonl/test.neutral.filter.jsonl"
MODEL_PATH="${MODEL_PATH:-pretrained_models/CosyVoice2-0.5B}"
# -----------------------------------------------------------------------

drop_prompts="1"
drop_targets="0"
cfg_scales="1.5 2.0 2.5 3.0"
filter_topks="-1"
cfg_rescales="1.0"

for drop_prompt in $drop_prompts; do
  for drop_target in $drop_targets; do
    if [[ "$drop_prompt" == "$drop_target" ]]; then
      echo "Skipping invalid combination: drop_prompt=$drop_prompt, drop_target=$drop_target (they must be different)"
      continue
    fi

    for cfg_scale in $cfg_scales; do
      for filter_topk in $filter_topks; do
        for cfg_rescale in $cfg_rescales; do
          if (( $(echo "$cfg_rescale >= $cfg_scale" | bc -l) )); then
            echo "Skipping: cfg_rescale ($cfg_rescale) >= cfg_scale ($cfg_scale)"
            continue
          fi

          if [[ "$filter_topk" == "-1" ]] && [[ "$cfg_rescale" != "1.0" ]]; then
            echo "Skipping: cfg_rescale ($cfg_rescale) not applicable when filter_topk ($filter_topk) is not > 0"
            continue
          fi

          output_audio=gen_audio_emotion_diff_reference/data_cfg_gpt_condition_emotion/cfg-$cfg_scale-drop_prompt_$drop_prompt-drop_target_$drop_target-topk_$filter_topk-rescale_$cfg_rescale
          output_dir=gen_audio_emotion_diff_reference/data_cfg_gpt_condition_emotion/cfg-$cfg_scale-drop_prompt_$drop_prompt-drop_target_$drop_target-topk_$filter_topk-rescale_$cfg_rescale-metadata
          output_dir_test=$output_dir/test
          export LOCAL_RANK=0
          {
            # Step 1: Generate audios with CFG
            python ./gen_audio_emotion_diff_reference/gen_dataset.py \
              --json_path "$TEST_JSONL" --model_path "$MODEL_PATH" \
              --target_emotion "$DATA_DIR/utt2target_emo" \
              --save_path $output_audio --cfg_scale $cfg_scale \
              --drop_prompt $drop_prompt --filter_topk $filter_topk \
              --rescale_cfg $cfg_rescale || exit 1;

            # Step 2: Convert to Kaldi format
            python "$EMOTION2VEC_DIR/convert_to_kaldi.py" --test_jsonl "$TEST_JSONL" --output_dir $output_dir
            find $PWD/$output_audio -type f -name "*.wav" > $output_dir_test/wavlist
            cat $output_dir_test/wavlist | rev | cut -d "/" -f1 | rev | sed 's/\.wav//g' | paste -d " " - $output_dir_test/wavlist > $output_dir_test/wav.scp

            # Step 3: emotion2vec accuracy
            python "$EMOTION2VEC_DIR/test_dataset.py" --wav_scp $output_dir_test/wav.scp --output_dir $output_dir_test
            python "$EMOTION2VEC_DIR/calculate_accuracy.py" --results $output_dir_test/emo_results.txt --ground_truth $output_dir_test/utt2emotion --output $output_dir_test/emo.acc

            # Step 4: UTMOS
            python "$UTMOS_DIR/predict.py" --ckpt_path "$UTMOS_CKPT" --mode predict_dir --inp_dir $output_audio --bs 200 --out_path $output_dir_test/utmos.tsv

            # Step 5: Whisper WER
            python ./scripts/get_wer.py --wav_scp $output_dir_test/wav.scp --output_result $output_dir_test/recog.result.whisperlv3 --reference $output_dir_test/text

            # Step 6: NISQA
            python "$NISQA_DIR/run_predict.py" --mode predict_wavscp --pretrained_model "$NISQA_DIR/weights/nisqa.tar" --wav_scp $output_dir_test/wav.scp --num_workers 2 --bs 100 --output_dir $output_dir_test
            python "$NISQA_DIR/run_predict.py" --mode predict_wavscp --pretrained_model "$NISQA_DIR/weights/nisqa_tts.tar" --wav_scp $output_dir_test/wav.scp --num_workers 2 --bs 100 --output_dir $output_dir_test

            # Step 7: DNSMOS
            python ./scripts/compute_dnsmos.py --mode wavscp --wav_scp $output_dir_test/wav.scp -o $output_dir_test --sample-rate 24000 --device "cuda:0"
          } &
        done
      done
    done
  done
done
