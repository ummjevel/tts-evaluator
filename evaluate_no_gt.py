from typing import Dict
import numpy as np
import torch
import whisper
import utmosv2
import os
import tempfile
import soundfile as sf

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer as jiwer_wer
from jiwer import cer as jiwer_cer

# ----------------------------
# Model Load
# ----------------------------

# UTMOSv2
# pip install git+https://github.com/sarulab-speech/UTMOSv2.git
utmosv2_model = utmosv2.create_model(pretrained=True)

# Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

whisper_model_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)

whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if torch.cuda.is_available() else -1,
)

# ----------------------------
# Compute Metrics
# ----------------------------

# UTMOSv2
def compute_utmos(gen_audio: np.ndarray, sr: int = 16000) -> float:
    """
    gen_audio: numpy array (1D or 2D waveform)
    sr: sample rate

    UTMOSv2는 파일 경로 입력을 받으므로, 임시 wav 파일로 저장 후 평가합니다.
    """
    # 모델 생성 (재사용하고 싶으면 전역변수로 빼는 게 효율적)
    global utmosv2_model

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
        # gen_audio가 2D (channels, samples)일 경우 처리
        if gen_audio.ndim == 2:
            audio_to_save = gen_audio.T  # (samples, channels)
        else:
            audio_to_save = gen_audio

        # 16kHz가 아니면 리샘플링 필요
        if sr != 16000:
            import torchaudio
            audio_to_save = torchaudio.functional.resample(
                torch.tensor(audio_to_save), orig_freq=sr, new_freq=16000
            ).numpy()

        # wav 파일로 저장
        sf.write(tmpfile.name, audio_to_save, samplerate=16000)

        # UTMOS 평가
        mos_score = utmosv2_model.predict(input_path=tmpfile.name)

    return float(mos_score)


# Whisper
def compute_asr_wer(audio: np.ndarray, sr: int, reference_text: str) -> float:
    """
    Whisper로 인식한 결과와 reference_text 비교 후 WER 반환
    """
    pred_text = _transcribe(audio, sr)
    ref_text = reference_text.lower().strip()
    return jiwer_wer(ref_text, pred_text)


def compute_asr_cer(audio: np.ndarray, sr: int, reference_text: str) -> float:
    """
    Whisper로 인식한 결과와 reference_text 비교 후 CER 반환
    """
    pred_text = _transcribe(audio, sr)
    ref_text = reference_text.lower().strip()
    return jiwer_cer(ref_text, pred_text)


def _transcribe(audio: np.ndarray, sr: int) -> str:
    """
    Whisper로 audio를 텍스트로 변환 (내부 사용 함수)
    """
    if sr != 16000:
        import torchaudio
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, 16000).numpy()
        sr = 16000

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
        sf.write(tmp_wav.name, audio, sr)
        result = asr_pipe(tmp_wav.name)

    pred_text = result["text"].lower().strip()
    return pred_text