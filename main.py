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
    parser.add_argument("--ref_texts", type=str, help="레퍼런스 텍스트")
    parser.add_argument("--save_path", type=str, help="결과 저장 경로")
    parser.add_argument("--save_asr_json_path", type=str, help="asr 결과 저장 경로")
    args = parser.parse_args()

    # load models once
    model_cache = load_models(device_id=4, language=args.language)

    # evaluator instance, set model_cache
    evaluator = Evaluator(sr=args.sr, language=args.language)
    evaluator.model_cache = model_cache

    evaluator.batch_evaluate(gen_dir=args.gen_dir, ref_dir=args.ref_dir, ref_texts=args.ref_texts
                            , save_path=args.save_path, save_asr_json_path=args.save_asr_json_path)  # 또는 ref 없이 단독 평가

