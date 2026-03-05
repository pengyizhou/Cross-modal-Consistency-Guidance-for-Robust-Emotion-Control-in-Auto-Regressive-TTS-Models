#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.


. ./path.sh || exit 1;

stage=-1
stop_stage=-1

# ====== Configuration (modify these) ======
pretrained_model_dir=$PWD/pretrained_models/CosyVoice2-0.5B
num_splits=4       # number of parallel jobs for feature extraction
num_gpus=1         # number of GPUs for training

# Directories for generated audio metadata (used in stage 0 and 1)
# These should contain ground-truth and generated audio metadata directories
gt_train_dir=""       # e.g., /path/to/gen_audio/data/ground_truth-metadata
gen_train_dirs=""     # e.g., /path/to/gen_audio/data/gen_instruct_gt_seed_{0,42}-metadata
gt_valid_dir=""       # e.g., /path/to/gen_audio/data_valid/ground_truth-metadata
gen_valid_dirs=""     # e.g., /path/to/gen_audio/data_valid/gen_instruct_gt_seed_{0,42}-metadata
utt_match_file=""     # e.g., /path/to/gen_audio/data/utt_Match

dpo_samples=data/dpo_samples
exp_dir=exp/dpo_training
# ===========================================

# Stage 0: Select DPO training samples
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Select DPO training samples"
  python ./tools/select_dpo_samples_v5.py \
    --inp_gt_dir ${gt_train_dir} \
    --out_dir ${dpo_samples} \
    --inp_gen_dirs ${gen_train_dirs} \
    --utt_match ${utt_match_file} \
    --disable_utmos \
    --disable_speed_warp \
    --disable_emo_syllable_dur \
    --disable_count_syllable_dur \
    --wers_min 0.80
fi

# Stage 1: Select DPO validation samples
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Select DPO validation samples"
  python ./tools/select_dpo_samples_v5.py \
    --inp_gt_dir ${gt_valid_dir} \
    --out_dir ${dpo_samples}_valid \
    --inp_gen_dirs ${gen_valid_dirs} \
    --disable_utmos \
    --disable_speed_warp \
    --disable_emo_syllable_dur \
    --disable_count_syllable_dur
fi

# Stage 5: Extract speech tokens and embeddings for validation set
dpo_folders="dpo_samples_valid"
for dpo_sample_folder in $dpo_folders; do
  if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    dpo_samples=data/${dpo_sample_folder}
    echo "Stage 5: Extract speech tokens and embeddings (validation)"
    for x in negative_samples positive_samples; do
        mkdir -p $dpo_samples/$x/
        tools/convert_to_kaldi.py --test_jsonl $dpo_samples/$x.jsonl --output_dir $dpo_samples/$x/
        mv $dpo_samples/$x/test/* $dpo_samples/$x/
        rm -rf $dpo_samples/$x/test $dpo_samples/$x/spk2utt
        cut -d " " -f1 < $dpo_samples/$x/wav.scp | awk '{print $1" "$1}' > $dpo_samples/$x/utt2spk
        ./utils/split_data.sh $dpo_samples/$x ${num_splits}
        for y in $(seq 1 ${num_splits}); do
          tools/extract_speech_token.py \
            --dir $dpo_samples/$x/split${num_splits}/$y \
            --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx &
        done
        wait

        for y in $(seq 1 ${num_splits}); do
          tools/extract_embedding.py \
            --dir $dpo_samples/$x/split${num_splits}/$y \
            --onnx_path $pretrained_model_dir/campplus.onnx &
        done
        wait
    done
  fi
done

# Stage 6: Extract speech tokens and embeddings for training set
dpo_folders="dpo_samples"
for dpo_sample_folder in $dpo_folders; do
  if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    dpo_samples=data/${dpo_sample_folder}
    echo "Stage 6: Extract speech tokens and embeddings (training)"
    for x in negative_samples positive_samples; do
        mkdir -p $dpo_samples/$x/
        tools/convert_to_kaldi.py --test_jsonl $dpo_samples/$x.jsonl --output_dir $dpo_samples/$x/
        mv $dpo_samples/$x/test/* $dpo_samples/$x/
        rm -rf $dpo_samples/$x/test $dpo_samples/$x/spk2utt
        cut -d " " -f1 < $dpo_samples/$x/wav.scp | awk '{print $1" "$1}' > $dpo_samples/$x/utt2spk
        ./utils/split_data.sh $dpo_samples/$x ${num_splits}
        for y in $(seq 1 ${num_splits}); do
          tools/extract_speech_token.py \
            --dir $dpo_samples/$x/split${num_splits}/$y \
            --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx &
        done
        wait

        for y in $(seq 1 ${num_splits}); do
          tools/extract_embedding.py \
            --dir $dpo_samples/$x/split${num_splits}/$y \
            --onnx_path $pretrained_model_dir/campplus.onnx &
        done
        wait
    done
  fi
done

# Stage 7: Combine split .pt files
dpo_folders="dpo_samples_valid dpo_samples"
pt_files="spk2embedding.pt utt2embedding.pt utt2speech_token.pt"
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Combine split .pt files"
    for dpo_sample_folder in $dpo_folders; do
      dpo_samples=data/${dpo_sample_folder}
      for x in negative_samples positive_samples; do
        for pt_file in $pt_files; do
          python tools/combine_pt.py $dpo_samples/$x/$pt_file $dpo_samples/$x/split${num_splits}/*/$pt_file
        done
      done
    done
fi

# Stage 8: Prepare parquet format data
dpo_folders="dpo_samples"
for dpo_sample_folder in $dpo_folders; do
  dpo_samples=data/${dpo_sample_folder}
  if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Prepare parquet data"
      echo "Required files: wav.scp, text, utt2spk, utt2embedding.pt, spk2embedding.pt, utt2speech_token.pt"
      mkdir -p $dpo_samples/dev $dpo_samples/train

      # Link speech tokens
      ln -sf $(realpath $dpo_samples/train/positive_samples/utt2speech_token.pt) $dpo_samples/train/utt2speech_token.pt
      ln -sf $(realpath $dpo_samples/train/negative_samples/utt2speech_token.pt) $dpo_samples/train/neg_utt2speech_token.pt
      ln -sf $(realpath $dpo_samples/dev/positive_samples/utt2speech_token.pt) $dpo_samples/dev/utt2speech_token.pt
      ln -sf $(realpath $dpo_samples/dev/negative_samples/utt2speech_token.pt) $dpo_samples/dev/neg_utt2speech_token.pt

      # Link embeddings (from your SFT dataset or precomputed embeddings)
      # Modify these paths to point to your embedding files:
      # ln -sf /path/to/sft_training/spk2embedding.pt $dpo_samples/train/spk2embedding.pt
      # ln -sf /path/to/sft_training/utt2embedding.pt $dpo_samples/train/utt2embedding.pt
      # ln -sf /path/to/sft_validation/spk2embedding.pt $dpo_samples/dev/spk2embedding.pt
      # ln -sf /path/to/sft_validation/utt2embedding.pt $dpo_samples/dev/utt2embedding.pt

      cp $dpo_samples/train/positive_samples/{text,style_prompts,wav.scp} $dpo_samples/train/
      cp $dpo_samples/dev/positive_samples/{text,style_prompts,wav.scp} $dpo_samples/dev/

      # Copy utt2spk files (modify paths as needed)
      # cp /path/to/your/train/utt2spk $dpo_samples/train/utt2spk
      # cp /path/to/your/valid/utt2spk $dpo_samples/dev/utt2spk

      for x in train dev; do
        mkdir -p $dpo_samples/$x/parquet
        tools/make_parquet_list_simple.py --num_utts_per_parquet 1000 \
                  --num_processes 10 \
                  --dpo \
                  --src_dir $dpo_samples/$x \
                  --des_dir $dpo_samples/$x/parquet
      done
  fi
done

# Stage 10: DPO Training
dpo_sample_folder="dpo_samples"
dpo_samples=data/${dpo_sample_folder}
exp_dir=exp/${dpo_sample_folder}
export CUDA_VISIBLE_DEVICES="0"
job_id=2007
port=2007
dist_backend="nccl"
num_workers=3
prefetch=100
train_engine=torch_ddp

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Stage 10: DPO training (LLM only)"
  model=llm
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:$port" \
    cosyvoice/bin/train_emotion.py \
    --train_engine $train_engine \
    --config conf/cosyvoice2_emo_prompt_dpo.yaml \
    --train_data $dpo_samples/train/parquet/data.list \
    --cv_data $dpo_samples/dev/parquet/data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --model $model \
    --checkpoint $pretrained_model_dir/$model.pt \
    --ref_model $pretrained_model_dir/llm.pt \
    --model_dir $exp_dir \
    --tensorboard_dir $exp_dir/$model/$train_engine/tensorboard \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --no_amp \
    --dpo
fi
