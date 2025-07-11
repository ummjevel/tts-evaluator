import os
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from typing import Dict
from tqdm import tqdm

from evaluator import TTSBatchEvaluator
from evaluate_no_gt import load_models
import os

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
    parser.add_argument("--device", type=int, default=0, help="결과 저장 경로")
    args = parser.parse_args()

    # load models once
    model_cache = load_models(device_id=args.device, language=args.language)

    # Whisper 모델과 UTMOS 모델 준비
    whisper = model_cache["fast_whisper_model"]
    utmos = model_cache["utmos_model"]  # 또는 이미 로드된 객체

    # 평가기 생성
    evaluator = TTSBatchEvaluator(whisper_model=whisper, utmos_model=utmos)

    # 평가 실행
    evaluator.batch_evaluate(
        gen_dir=args.gen_dir,
        ref_dir=args.ref_dir,
        ref_texts=args.ref_texts,
        save_path=args.save_path,
        save_asr_json_path=args.save_asr_json_path
    )