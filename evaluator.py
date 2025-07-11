import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from typing import Dict, Callable, Optional, Union, Tuple
from evaluate_with_gt import compute_pesq, compute_stoi, compute_mcd, compute_snr, compute_lsd, compute_f0_rmse, compute_periodicity_f1
from evaluate_no_gt import compute_utmos, compute_asr_wer, compute_asr_cer, compute_asr_cer_fasterwhisper, compute_asr_cer_fasterwhisper_batch, compute_utmos_batch

import json
from datetime import datetime
import tempfile

METRICS_NO_GT = {
    "UTMOS": compute_utmos,
    "ASR_Whisper": [compute_asr_wer, compute_asr_cer_fasterwhisper],
}

METRICS_WITH_GT = {
    "PESQ": compute_pesq,
    "STOI": compute_stoi,
    "MCD": compute_mcd,
    "SNR": compute_snr,
    "LSD": compute_lsd,
    "F0_RMSE": compute_f0_rmse,
    "Periodicity_F1": compute_periodicity_f1,
}


class Evaluator:
    def __init__(self, sr: int = 16000, language: str = "en", cache_dir: Optional[str] = None):
        self.sr = sr
        self.language = language.lower()
        
        # 캐시 폴더 경로 설정
        if cache_dir is None:
            self.cache_dir = tempfile.gettempdir()  # 시스템 임시 디렉토리 기본값
        else:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)

    def save_temp_wav(self, audio: np.ndarray, filename_prefix: str = "temp_audio") -> str:
        temp_path = os.path.join(self.cache_dir, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
        sf.write(temp_path, audio, self.sr)
        return temp_path

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

                if "ASR_Whisper" in name and self.language.lower() in ["ko", "korean"]:
                    results[name], results["ASR_Pred"] = selected_func(gen_audio, self.sr, ref_text, model=self.model_cache["fast_whisper_model"])
                    results["ASR_Ref"] = ref_text
                elif "ASR_Whisper" in name:
                    results[name], results["ASR_Pred"] = selected_func(gen_audio, self.sr, ref_text, model=self.model_cache["asr_model"])
                    results["ASR_Ref"] = ref_text
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


    def batch_evaluate(self, gen_dir: str, ref_dir: Optional[str] = None
                , ext: str = "wav", ref_texts: Optional[str] = None
                , save_path: Optional[str] = None
                , save_asr_json_path: Optional[str] = None):
 
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

            base_fname = fname
            if "_" in fname:
                base_fname = fname.split("_")[0] + ".wav"
            else:
                base_fname = fname

            if ref_path is not None:
                ref_path = os.path.join(ref_dir, base_fname)

            ref_text = text_map.get(base_fname, "")
            result = self.evaluate_file(gen_path, ref_path, ref_text=ref_text)

            results.append(result)

        # 3. 결과 평균 출력
        keys = [k for k in results[0].keys() if k != "file"]
        print("\n=== 평균 결과 ===")
        avg_result = {}
        for key in keys:
            vals = [r[key] for r in results if r[key] is not None]
            # 🔒 숫자 값만 평균 처리
            if all(isinstance(v, (int, float, np.number)) for v in vals):
                avg_val = np.mean(vals) if vals else None
                avg_result[key] = avg_val
                print(f"{key}: {avg_val:.4f}" if avg_val is not None else f"{key}: None (all missing)")
            else:
                # 숫자가 아닌 경우는 저장하지 않거나 따로 처리 가능
                pass

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

        # Whisper 결과 포함 JSON 저장
        if save_asr_json_path:
            with open(save_asr_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n📁 평가 결과가 JSON으로 저장되었습니다: {save_path}")




class TTSBatchEvaluator:
    def __init__(self, whisper_model, utmos_model):
        self.whisper_model = whisper_model
        self.utmos_model = utmos_model

    def batch_evaluate(
        self,
        gen_dir: str,
        ref_dir: Optional[str] = None,  # 사용하지 않지만 인터페이스 유지
        ref_texts: Optional[str] = None,
        save_path: Optional[str] = None,
        save_asr_json_path: Optional[str] = None,
    ):
        if ref_texts is None:
            raise ValueError("ref_texts must be provided for CER evaluation.")

        # 1. CER 평가
        asr_results = compute_asr_cer_fasterwhisper_batch(
            folder_path=gen_dir,
            txt_path=ref_texts,
            model=self.whisper_model,
        )

        # 2. UTMOS 평가
        utmos_scores = compute_utmos_batch(
            gen_dir=gen_dir,
            sr=24000,
            model=self.utmos_model,
        )

        # 3. 결과 통합 (같은 순서대로 매핑)
        if len(asr_results) != len(utmos_scores):
            raise ValueError("asr_results와 utmos_scores의 길이가 다릅니다!")

        for item, utmos in zip(asr_results, utmos_scores):
            item["UTMOS"] = utmos

        # 4. 평균 요약
        print("\n📊 평균 결과:")
        summary = {
            "UTMOS": np.mean([r for r in utmos_scores]),
            "ASR_Whisper_CER": np.mean([r["ASR_Whisper_CER"] for r in asr_results]),
        }
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")

        # 5. 저장
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(gen_dir, f"eval_results_{timestamp}.json")

        output = {
            "summary": summary,
            "details": asr_results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 평가 결과가 저장되었습니다 → {save_path}")

        if save_asr_json_path:
            with open(save_asr_json_path, "w", encoding="utf-8") as f:
                json.dump(asr_results, f, indent=2, ensure_ascii=False)
            print(f"📁 ASR 평가 결과가 저장되었습니다 → {save_asr_json_path}")