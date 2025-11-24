"""
Usage:
python extract_speecht5.py \
  --dir a \
  --model microsoft/speecht5_vc \
  --out_file feature.h5
"""
import argparse
import json
import os
import sys
from pathlib import Path
import librosa
import torchaudio

import numpy as np
import torch

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech

from tqdm import tqdm
import h5py

def extract(dir_path, model_name_or_path, out_file, batch_size=1):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    device = "cpu"

    processor = SpeechT5Processor.from_pretrained(model_name_or_path)
    model = SpeechT5ForSpeechToSpeech.from_pretrained(model_name_or_path).to(device).eval()

    wav_set = set(Path(dir_path).glob("*.wav"))
    wav_set = sorted(list(wav_set))

    with h5py.File(out_file, "w") as h5f:      
        for i in tqdm(range(0, len(wav_set), batch_size), desc="Extract"):
            batch_samples = wav_set[i : i + batch_size]
            wavs = []
            keys = []
            for s in batch_samples:
                wav, sr = torchaudio.load(s)  # wav: Tensor (C, T)
                if sr != 16_000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                    sr = 16000
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)  # (1, T)
                wav_np = wav.squeeze(0).cpu().numpy()
                wavs.append(wav_np)
                keys.append(s)  # keep original path for key derivation

            # print("=== debug shapes ===")
            # print("type(wavs[0])", type(wavs[0]))            # 看看单个 wav 的类型
            # print("len(wavs) (batch)", len(wavs))

            lengths = [len(w) for w in wavs]            # 每条波形的样本数
            # print("batch size", len(wavs), ", lengths(samples):", lengths)
            inputs = processor(audio=wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            # print("input_values shape:", inputs["input_values"].shape)          
            # if "attention_mask" in inputs:
            #     print("attention_mask shape:", inputs["attention_mask"].shape)  
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                x = model.speecht5.encoder(**inputs)
                enc = x.last_hidden_state  # (B, T_max, D)

            attn_mask = inputs.get("attention_mask", None) 
            if attn_mask is not None:
                attn_mask = attn_mask.cpu() 
            enc = enc.cpu()  

            for j, s in enumerate(batch_samples):
                key = Path(s).stem
                if key in h5f:
                    continue

                if attn_mask is not None:
                    valid_len = int(attn_mask[j].sum().item())  # number of valid frames
                    vec = enc[j, :valid_len, :].numpy().astype(np.float32)  # shape (T_i, D)
                else:
                    vec = enc[j].numpy().astype(np.float32)

                # print(vec.shape)
                h5f.create_dataset(key, data=vec, dtype=np.float32)

    print(f"Saved {len(wav_set)} vectors → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="folder containing 16 kHz wavs")
    parser.add_argument("--model", default="microsoft/speecht5_vc")
    parser.add_argument("--out_file", default="./speecht5_emb.h5")
    args = parser.parse_args()
    extract(args.dir, args.model, args.out_file)