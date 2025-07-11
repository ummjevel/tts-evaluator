import numpy as np
import torch
import tempfile
import torchaudio
import soundfile as sf
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import utmosv2
from faster_whisper import WhisperModel
from typing import Dict, List
import os

def load_models(device_id: int = 0, cache_dir: str = None, language: str = None):
    """
    모델을 한 번만 로드하는 함수.

    Args:
        device_id (int): 사용할 GPU ID. CPU 사용 시 -1.
        cache_dir (str, optional): Hugging Face 모델 캐시 경로. 기본값 None.

    Returns:
        dict: {
            "utmos_model": utmosv2 모델,
            "asr_model": Whisper 파이프라인
        }
    """
    # UTMOSv2
    utmos_model = utmosv2.create_model(pretrained=True)

    # Whisper
    use_cuda = device_id >= 0 and torch.cuda.is_available()
    device = f"cuda:{device_id}" if use_cuda else "cpu"

    torch_dtype = torch.float16 if use_cuda else torch.float32

    whisper_model_id = "openai/whisper-large-v3"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=cache_dir,
    ).to(device)

    whisper_processor = AutoProcessor.from_pretrained(
        whisper_model_id,
        cache_dir=cache_dir,
    )

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device_id if use_cuda else -1,
    )

    device_asr = f"cuda" if use_cuda else "cpu"
    fast_whisper_model = WhisperModel("large-v3", device=device_asr, compute_type="float16")

    return {
        "utmos_model": utmos_model,
        "asr_model": asr_pipe,
        "fast_whisper_model": fast_whisper_model
    }


def compute_utmos(gen_audio: np.ndarray, sr: int, model=None) -> float:
    if model is None:
        raise ValueError("UTMOS model is not provided.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
        if gen_audio.ndim == 2:
            gen_audio = gen_audio.T
        if sr != 16000:
            gen_audio = torchaudio.functional.resample(torch.tensor(gen_audio), sr, 16000).numpy()
        sf.write(tmpfile.name, gen_audio, samplerate=16000)
        mos_score = model.predict(input_path=tmpfile.name)

    return float(mos_score)


def compute_utmos_batch(gen_dir: str, sr: int, model=None) -> float:
    if model is None:
        raise ValueError("UTMOS model is not provided.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        valid_filenames = []
        for fname in sorted(os.listdir(gen_dir)):
            if not fname.endswith(".wav"):
                continue
            file_path = os.path.join(gen_dir, fname)

            # 오디오 로드 및 채널/샘플레이트 확인
            audio, sr = torchaudio.load(file_path)

            # 스테레오 -> 모노 변환
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # 샘플레이트 변경
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                sr = 16000

            # 저장 (1D로 저장)
            output_path = os.path.join(tmp_dir, fname)
            sf.write(output_path, audio.squeeze().numpy(), samplerate=sr)

            valid_filenames.append(output_path)

        # 예측 수행
        mos_score = model.predict(input_dir=tmp_dir)

        print(mos_score)
        # 정렬된 파일명 순으로 점수 리스트 생성
        path_to_score = {
            item["file_path"]: item["predicted_mos"]
            for item in mos_score
        }

        # 정렬된 파일 순서로 점수만 추출
        utmos_scores = [path_to_score[f] for f in valid_filenames if f in path_to_score]
        print(utmos_scores)
    return utmos_scores


def compute_asr_wer(audio: np.ndarray, sr: int, ref_text: str, model=None) -> tuple[float, str]:
    pred_text = _transcribe(audio, sr, model)
    return jiwer_wer(ref_text.lower().strip(), pred_text), pred_text


def compute_asr_cer(audio: np.ndarray, sr: int, ref_text: str, model=None) -> tuple[float, str]:
    pred_text = _transcribe(audio, sr, model)
    return jiwer_cer(ref_text.lower().strip(), pred_text), pred_text


def compute_asr_cer_fasterwhisper(audio: np.ndarray, sr: int, ref_text: str, model: WhisperModel) -> tuple[float, str]:
    """
    faster-whisper 기반 CER 계산

    Args:
        audio (np.ndarray): 오디오 waveform (1D or 2D).
        sr (int): 오디오 샘플링 레이트.
        ref_text (str): 레퍼런스 텍스트.
        model (WhisperModel): faster-whisper 모델 인스턴스.

    Returns:
        cer (float): 문자 오류율 (CER)
        pred_text (str): Whisper로부터 예측된 텍스트
    """
    if audio.ndim == 2:  # stereo
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        import torchaudio
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, 16000).numpy()
        sr = 16000

    # 임시 파일로 저장 후 디코딩 (faster-whisper는 파일이나 배열 모두 지원)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
        sf.write(tmpfile.name, audio, sr)
        segments, _info = model.transcribe(tmpfile.name, beam_size=5)
        pred_text = " ".join([seg.text for seg in segments]).strip()

    cer = jiwer_cer(ref_text.lower().strip(), pred_text)
    return cer, pred_text


def load_ref_dict_from_txt(txt_path: str) -> Dict[str, str]:
    """
    텍스트 파일로부터 {파일명: 참조문장} 딕셔너리 생성

    파일 형식 예시:
    SPK001/SPK001XXXXF001.wav|정답 문장|SPK001
    """
    ref_dict = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) < 2:
                continue
            path = parts[0]
            text = parts[1]
            fname = os.path.basename(path)
            ref_dict[fname] = text.strip()
    return ref_dict


def compute_asr_cer_fasterwhisper_batch(
    folder_path: str,
    txt_path: str,
    model: WhisperModel,
    beam_size: int = 5
) -> List[Dict]:
    """
    폴더 내 모든 .wav 파일에 대해 CER 평가 수행 (참조문장은 txt에서 추출)

    Args:
        folder_path (str): .wav 파일들이 있는 폴더
        txt_path (str): 참조 문장이 포함된 .txt 파일 경로
        model (WhisperModel): faster-whisper 모델
        beam_size (int): 디코딩 beam size

    Returns:
        List[Dict]: CER 평가 결과 리스트
    """
    ref_dict = load_ref_dict_from_txt(txt_path)
    results = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, fname)

        # 파일명에서 확장자 제거
        base_name = os.path.splitext(fname)[0]  # e.g. 'somefile_0'

        # _0 접미사 제거 시도
        if base_name.endswith("_0"):
            base_name_no_suffix = base_name[:-2]  # 'somefile'
        else:
            base_name_no_suffix = base_name

        # ref_dict 키 체크
        if fname in ref_dict:
            ref_key = fname
        elif base_name_no_suffix + ".wav" in ref_dict:
            ref_key = base_name_no_suffix + ".wav"
        else:
            print(f"[WARN] Reference not found for {fname}")
            continue

        # 오디오 로드
        audio, sr = torchaudio.load(file_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            sf.write(tmpfile.name, audio.squeeze().numpy(), 16000)
            segments, _info = model.transcribe(tmpfile.name, beam_size=beam_size)
            pred_text = " ".join([seg.text for seg in segments]).strip()

        ref_text = ref_dict[ref_key].strip().lower()
        cer = jiwer_cer(ref_text, pred_text)

        results.append({
            "file": ref_key,
            "ASR_Pred": pred_text,
            "ASR_Ref": ref_text,
            "ASR_Whisper_CER": cer
        })

    return results


def _transcribe(audio: np.ndarray, sr: int, model) -> str:
    if model is None:
        raise ValueError("ASR pipeline (Whisper) is not provided.")

    if sr != 16000:
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, 16000).numpy()
        sr = 16000

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        sf.write(tmp_wav.name, audio, sr)
        result = model(tmp_wav.name)

    return result["text"].lower().strip()
