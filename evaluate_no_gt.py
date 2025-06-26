import numpy as np
import torch
import tempfile
import torchaudio
import soundfile as sf
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import utmosv2


def load_models(device_id: int = 0):
    """
    모델을 한 번만 로드하는 함수.
    
    Args:
        device_id (int): 사용할 GPU ID. CPU 사용 시 -1.
        
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
    ).to(device)

    whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device_id if use_cuda else -1,
    )

    return {
        "utmos_model": utmos_model,
        "asr_model": asr_pipe,
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


def compute_asr_wer(audio: np.ndarray, sr: int, ref_text: str, model=None) -> float:
    pred_text = _transcribe(audio, sr, model)
    return jiwer_wer(ref_text.lower().strip(), pred_text)


def compute_asr_cer(audio: np.ndarray, sr: int, ref_text: str, model=None) -> float:
    pred_text = _transcribe(audio, sr, model)
    return jiwer_cer(ref_text.lower().strip(), pred_text)


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
