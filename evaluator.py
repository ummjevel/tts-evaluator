import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from typing import Dict, Callable, Optional, Union, Tuple
from evaluate_with_gt import compute_pesq, compute_stoi, compute_mcd, compute_snr, compute_lsd, compute_f0_rmse, compute_periodicity_f1
from evaluate_no_gt import compute_utmos, compute_mosnet, compute_asr_wer

import json
from datetime import datetime


METRICS_NO_GT = {
    "UTMOS": compute_utmos,
    "ASR_Whisper": [compute_asr_wer, compute_asr_cer],
}

METRICS_WITH_GT = {
    "MCD": compute_mcd,
    "SNR": compute_snr,
    "LSD": compute_lsd,
    "F0_RMSE": compute_f0_rmse,
    "Periodicity_F1": compute_periodicity_f1,
}


class Evaluator:
    def __init__(self, sr: int = 16000, language: str = "EN"):
        self.sr = sr
        self.language = language
        # 필요한 모델 캐시 or 초기화
        self.model_cache = {
            "utmos_model": self._load_utmos_model(),
            "mosnet_model": self._load_mosnet_model(),
            "asr_model": self._load_asr_model()
        }

    def _load_utmos_model(self):
        # UTMOS 모델 로드
        return None

    def _load_mosnet_model(self):
        return None

    def _load_asr_model(self):
        return None

    def evaluate_with_gt(self, ref_audio: np.ndarray, gen_audio: np.ndarray) -> Dict[str, float]:
        min_len = min(len(ref_audio), len(gen_audio))
        ref_audio, gen_audio = ref_audio[:min_len], gen_audio[:min_len]

        results = {}
        for name, func in METRICS_WITH_GT.items():
            try:
                results[name] = func(ref_audio, gen_audio, self.sr)
            except Exception as e:
                print(f"[Warning] {name} 실패: {e}")
                results[name] = None
        return results

    # evaluator 클래스 내부
    def evaluate_no_gt(self, gen_audio: np.ndarray, ref_text="") -> Dict[str, float]:
        results = {}
        for metric_name, metric_func in METRICS_NO_GT.items():
            try:
                if isinstance(metric_func, list):
                    if self.language.lower() in ["ko", "korean", "zh", "chinese", "ja", "japanese"]:
                        selected_func = metric_func[1]  # CER
                        name = f"{metric_name}_CER"
                    else:
                        selected_func = metric_func[0]  # WER
                        name = f"{metric_name}_WER"
                else:
                    selected_func = metric_func
                    name = metric_name

                if "ASR_Whisper" in name:
                    results[name] = selected_func(gen_audio, self.sr, ref_text, model=self.model_cache["asr_model"])
                else:
                    results[name] = selected_func(gen_audio, self.sr, model=self.model_cache["utmos_model"])
            except Exception as e:
                print(f"[Warning] {name} 실패: {e}")
                results[name] = None
        return results


    def evaluate_file(self, gen_path: str, ref_path: Optional[str] = None, ref_text: str = "") -> Dict[str, float]:
        gen_audio, sr_gen = sf.read(gen_path)
        if sr_gen != self.sr:
            gen_audio = torchaudio.functional.resample(torch.tensor(gen_audio), sr_gen, self.sr).numpy()

        if ref_path:
            ref_audio, sr_ref = sf.read(ref_path)
            if sr_ref != self.sr:
                ref_audio = torchaudio.functional.resample(torch.tensor(ref_audio), sr_ref, self.sr).numpy()
            result = self.evaluate_with_gt(ref_audio, gen_audio)
        else:
            result = self.evaluate_no_gt(gen_audio, ref_text=ref_text)

        result["file"] = os.path.basename(gen_path)
        return result


    def batch_evaluate(
        self,
        gen_dir: str,
        ref_dir: Optional[str] = None,
        ext: str = "wav",
        ref_texts: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        # 1. ref_texts 로부터 텍스트 맵 로딩
        text_map = {}
        if ref_texts:
            with open(ref_texts, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        wav_path, text = parts[0], parts[1]
                        fname = os.path.basename(wav_path)
                        text_map[fname] = text

        # 2. 오디오 파일 목록 추출
        files = [f for f in sorted(os.listdir(gen_dir)) if f.endswith(f".{ext}")]
        results = []

        for fname in tqdm(files):
            gen_path = os.path.join(gen_dir, fname)
            ref_path = os.path.join(ref_dir, fname) if ref_dir else None
            ref_text = text_map.get(fname, "")
            result = self.evaluate_file(gen_path, ref_path, ref_text=ref_text)
            results.append(result)

        # 3. 결과 평균 출력
        keys = [k for k in results[0].keys() if k != "file"]
        print("\n=== 평균 결과 ===")
        avg_result = {}
        for key in keys:
            vals = [r[key] for r in results if r[key] is not None]
            if vals:
                avg = np.mean(vals)
                print(f"{key}: {avg:.4f}")
                avg_result[key] = round(float(avg), 4)
            else:
                print(f"{key}: None (all missing)")
                avg_result[key] = None

        # 4. 결과 JSON으로 저장
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(gen_dir, f"eval_results_{timestamp}.json")

        output = {
            "summary": avg_result,
            "details": results
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n📁 평가 결과가 JSON으로 저장되었습니다: {save_path}")

