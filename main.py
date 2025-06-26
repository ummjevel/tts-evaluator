import os
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Dict
from tqdm import tqdm

from evaluator import Evaluator
from evaluate_no_gt import load_models

# ----------------------------
# Main (argparse)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="합성 음성 폴더")
    parser.add_argument("--ref_dir", type=str, default=None, help="GT 음성 폴더 (없으면 no-GT 평가)")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--language", type=str, default='ko', help="Language code (e.g. EN, KR, ZH)")
    args = parser.parse_args()

    # load models once
    model_cache = load_models(device_id=5)

    # evaluator instance, set model_cache
    evaluator = Evaluator(sr=args.sr, language=args.language)
    evaluator.model_cache = model_cache

    # no gt
    evaluator.batch_evaluate(args.gen_dir)  # 또는 ref 없이 단독 평가
    # with gt
    # evaluator.batch_evaluate(args.gen_dir, args.ref_dir)  # 또는 ref 없이 단독 평가
