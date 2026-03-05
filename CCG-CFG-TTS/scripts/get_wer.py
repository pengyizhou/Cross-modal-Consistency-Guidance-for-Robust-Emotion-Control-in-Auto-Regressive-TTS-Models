#!/usr/bin/env python3

import whisperx
import argparse
import subprocess
import os
import sys
import time
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

def init_asrModel():
    asr_model = whisperx.load_model("large-v3-turbo", device="cuda", device_index=0)
    return asr_model

def transcribe_audio(asr_model, audio_paths):
    transcriptions = {}
    total_files = len(audio_paths)
    
    # Create a fancy progress bar with additional information
    progress_bar = tqdm(
        audio_paths.items(), 
        total=total_files,
        desc="Transcribing audio files",
        unit="file",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for utt_id, audio_path in progress_bar:
        # Update progress bar description with current file
        progress_bar.set_description(f"Processing: {Path(audio_path).name}")
        
        # Convert string path to Path object to handle whitespace
        audio_path = Path(audio_path)
        audio = whisperx.load_audio(str(audio_path))
        transcription = asr_model.transcribe(audio, batch_size=4, language="en")
        transcriptions[utt_id] = " ".join(each.get("text") for each in transcription.get("segments"))
        
    return transcriptions

def get_wer(ref, hyp):
    compute_wer_script = "./scripts/compute-wer.py"
    
    # Run compute-wer.py with character-level analysis
    ref_file = ref
    hyp_file = hyp
    wer_file = hyp + ".wer"

    cmd = [
        "python", compute_wer_script,
        "--char=1",  # Character-level analysis
        "--v=1",     # Verbose output
        ref_file,
        hyp_file
    ]
    
    print(f"🧮 Calculating WER metrics...")
    print(f"   - Reference: {ref_file}")
    print(f"   - Hypothesis: {hyp_file}")
    print(f"   - Output: {wer_file}")
    
    # Create a spinner for the WER calculation
    with tqdm(total=1, desc="Computing WER", bar_format="{l_bar}{bar}| [{elapsed}]") as pbar:
        # Run the command and redirect output to wer file
        with open(wer_file, "w") as wer_output:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=wer_output,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                pbar.update(1)
                print(f"✅ WER analysis completed and saved to {wer_file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error calculating WER: {e.stderr}")
                raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True, type=str, help="path to wav.scp")
    parser.add_argument("--output_result", required=True, type=str, help="path to output result")
    parser.add_argument("--reference", required=True, type=str, help="path to reference transcript")
    args = parser.parse_args()

    print(f"📋 Loading ASR model (large-v3-turbo)...")
    asr_model = init_asrModel()
    print(f"✅ Model loaded successfully!")
    
    audio_paths = dict()
    # Load audio paths with a progress spinner
    print(f"🔍 Reading audio paths from {args.wav_scp}")
    with open(args.wav_scp, 'r') as f:
        wav_lines = f.readlines()
        
    # Show a progress bar for loading paths
    for wav in tqdm(wav_lines, desc="Loading audio paths", unit="file"):
        utt_id, path = wav.replace("\n", "").strip().split(maxsplit=1)  # Use maxsplit=1 to handle spaces in paths
        audio_paths[utt_id] = path

    transcriptions = transcribe_audio(asr_model, audio_paths)
    
    print(f"📝 Writing transcriptions to {args.output_result}")
    with open(args.output_result, 'w') as f:
        for utt_id, transcription in tqdm(transcriptions.items(), desc="Writing results", unit="transcription"):
            f.write(f"{utt_id} {transcription}\n")
    
    print(f"📊 Calculating WER against reference {args.reference}")        
    get_wer(args.reference, args.output_result)
    print(f"🎉 All processing complete! Check WER results in {args.output_result}.wer")
    

if __name__ == "__main__":
    start_time = time.time()
    try:
        print(f"🚀 Starting WER evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main()
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"✨ Processing completed in {int(minutes)}m {int(seconds)}s")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)