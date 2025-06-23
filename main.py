import os
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Dict
from tqdm import tqdm
from evaluator import batch_evaluate

# ----------------------------
# Main (argparse)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="합성 음성 폴더")
    parser.add_argument("--ref_dir", type=str, default=None, help="GT 음성 폴더 (없으면 no-GT 평가)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--ext", type=str, default="wav")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g. EN, KR, ZH)")
    args = parser.parse_args()

    batch_evaluate(args.gen_dir, args.ref_dir, args.sr, args.ext, args.language)