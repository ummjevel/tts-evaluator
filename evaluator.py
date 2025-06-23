import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from typing import Dict, Callable, Optional
from evaluate_with_gt import compute_pesq, compute_stoi, compute_mcd, compute_snr, compute_lsd, compute_f0_rmse, compute_periodicity_f1
from evaluate_no_gt import compute_utmos, compute_mosnet, compute_asr_wer

# ----------------------------
# Evaluation with GT (ref + gen)
# ----------------------------
# metric registry 딕셔너리
METRICS_WITH_GT: Dict[str, Callable] = {
    "PESQ": compute_pesq,
    "STOI": compute_stoi,
    "MCD": compute_mcd,
    "SNR": compute_snr,
    "LSD": compute_lsd,
    "F0_RMSE": compute_f0_rmse,
    "Periodicity_F1": compute_periodicity_f1,
}

def evaluate_with_gt(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    # 길이 맞추기
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio = ref_audio[:min_len]
    gen_audio = gen_audio[:min_len]

    results = {}
    for metric_name, metric_func in METRICS_WITH_GT.items():
        try:
            results[metric_name] = metric_func(ref_audio, gen_audio, sr)
        except Exception as e:
            print(f"[Warning] {metric_name} 계산 실패: {e}")
            results[metric_name] = None
    return results


# ----------------------------
# Evaluation without GT (gen only)
# ----------------------------
# metric registry 딕셔너리
METRICS_NO_GT: Dict[str, Union[Callable, list]] = {
    "UTMOS": compute_utmos,
    "ASR_Whisper": [compute_asr_wer, compute_asr_cer]
}

def evaluate_no_gt(gen_audio: np.ndarray, sr: int = 16000, language: str = "EN") -> Dict[str, float]:
    results = {}
    for metric_name, metric_func in METRICS_NO_GT.items():
        try:
            if isinstance(metric_func, list):
                # 언어에 따라 WER or CER 결정
                if language.lower() in ["ko", "korean", "zh", "chinese", "ja", "japanese"]:
                    selected_func = metric_func[1]  # CER
                    metric_name = f"{metric_name}_CER"
                else:
                    selected_func = metric_func[0]  # WER
                    metric_name = f"{metric_name}_WER"
            else:
                selected_func = metric_func

            results[metric_name] = selected_func(gen_audio, sr)
        except Exception as e:
            print(f"[Warning] {metric_name} 계산 실패: {e}")
            results[metric_name] = None
    return results


# ----------------------------
# Evaluate one file (pair or single)
# ----------------------------
def evaluate_file(gen_path: str, ref_path: Optional[str] = None, sr: int = 16000, language: str = "EN") -> Dict[str, float]:
    gen_audio, sr_gen = sf.read(gen_path)
    if sr_gen != sr:
        gen_audio = torchaudio.functional.resample(torch.tensor(gen_audio), sr_gen, sr).numpy()
    
    if ref_path:
        ref_audio, sr_ref = sf.read(ref_path)
        if sr_ref != sr:
            ref_audio = torchaudio.functional.resample(torch.tensor(ref_audio), sr_ref, sr).numpy()
        return evaluate_with_gt(ref_audio, gen_audio, sr)
    else:
        return evaluate_no_gt(gen_audio, sr, language)


# ----------------------------
# Batch evaluate
# ----------------------------
def batch_evaluate(gen_dir: str, ref_dir: Optional[str] = None, sr: int = 16000, ext: str = "wav", language: str = "EN"):
    results = []
    for fname in tqdm(sorted(os.listdir(gen_dir))):
        if not fname.endswith(f".{ext}"):
            continue
        gen_path = os.path.join(gen_dir, fname)
        ref_path = os.path.join(ref_dir, fname) if ref_dir is not None else None
        metrics = evaluate_file(gen_path, ref_path, sr)
        metrics["file"] = fname
        results.append(metrics)

    if len(results) == 0:
        print("No files found for evaluation.")
        return

    # 공통 metric key 추출 (file 제외)
    all_keys = [k for k in results[0].keys() if k != "file"]

    print("\n=== 평균 결과 ===")
    for key in all_keys:
        vals = [r[key] for r in results if r[key] is not None]
        if len(vals) > 0:
            avg = np.mean(vals)
            print(f"{key}: {avg:.4f}")
        else:
            print(f"{key}: None (all values missing)")
