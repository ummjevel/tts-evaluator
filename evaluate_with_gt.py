import numpy as np
import torch
import torchaudio
import torchcrepe
from python_speech_features import mfcc
from scipy.spatial.distance import euclidean
import librosa
from torchcrepe.loudness import REF_DB
from pesq import pesq
from pystoi import stoi

SILENCE_THRESHOLD = -60
UNVOICED_THRESHOLD = 0.21

# ----------------------------
# Metric Functions
# ----------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    오디오 리샘플링 (scipy 사용)
    """
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down)

def compute_pesq(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000) -> float:
    """
    PESQ 계산 (ITU-T P.862)
    sr은 8000 또는 16000만 지원 (nb 또는 wb).
    다른 sr이 들어오면 16kHz로 리샘플링 후 wide-band 평가.
    """
    # 리샘플링 필요 여부 판단
    if sr not in (8000, 16000):
        target_sr = 16000
        ref_audio = _resample(ref_audio, sr, target_sr)
        gen_audio = _resample(gen_audio, sr, target_sr)
        sr = target_sr

    mode = "wb" if sr == 16000 else "nb"
    return pesq(sr, ref_audio, gen_audio, mode)

def compute_stoi(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000) -> float:
    """
    STOI 계산
    """
    return stoi(ref_audio, gen_audio, sr, extended=False)

def compute_mcd(ref: np.ndarray, gen: np.ndarray, sr: int) -> float:
    ref_mfcc = mfcc(ref, sr)
    gen_mfcc = mfcc(gen, sr)
    length = min(len(ref_mfcc), len(gen_mfcc))
    return np.mean([euclidean(r, d) for r, d in zip(ref_mfcc[:length], gen_mfcc[:length])])


def compute_snr(ref: np.ndarray, gen: np.ndarray, sr: int = None) -> float:
    noise = ref - gen
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum(noise ** 2) + 1e-8))


def compute_lsd(ref: np.ndarray, gen: np.ndarray, sr: int) -> float:
    spec_ref = np.abs(np.fft.rfft(ref))
    spec_gen = np.abs(np.fft.rfft(gen))
    return np.mean(np.sqrt((20 * np.log10((spec_ref + 1e-8)/(spec_gen + 1e-8)))**2))


def compute_f0_rmse(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int) -> float:
    ref_tensor = torch.tensor(ref_audio).unsqueeze(0)
    gen_tensor = torch.tensor(gen_audio).unsqueeze(0)
    pitch_ref, _ = torchaudio.functional.detect_pitch_frequency(ref_tensor, sr)
    pitch_gen, _ = torchaudio.functional.detect_pitch_frequency(gen_tensor, sr)
    return torch.sqrt(torch.mean((pitch_ref - pitch_gen) ** 2)).item()


def compute_periodicity_f1(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int) -> float:
    y = torch.tensor(ref_audio).unsqueeze(0)
    y_hat = torch.tensor(gen_audio).unsqueeze(0)

    true_pitch, true_periodicity = predict_pitch(y)
    pred_pitch, pred_periodicity = predict_pitch(y_hat)

    true_voiced = ~np.isnan(true_pitch)
    pred_voiced = ~np.isnan(pred_pitch)

    # voiced/unvoiced F1 score
    true_positives = (true_voiced & pred_voiced).sum()
    false_positives = (~true_voiced & pred_voiced).sum()
    false_negatives = (true_voiced & ~pred_voiced).sum()

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return float(f1)


# ----------------------------
# Internal - CREPE 기반 pitch/periodicity 예측
# ----------------------------
"""
Periodicity metrics adapted from Vocos repository.(https://github.com/gemelo-ai/vocos)
Vocos adapted from https://github.com/descriptinc/cargan
"""

def predict_pitch(
    audio: torch.Tensor, silence_threshold: float = SILENCE_THRESHOLD, unvoiced_treshold: float = UNVOICED_THRESHOLD
):
    """
    Predicts pitch and periodicity for the given audio.

    Args:
        audio (Tensor): The audio waveform.
        silence_threshold (float): The threshold for silence detection.
        unvoiced_treshold (float): The threshold for unvoiced detection.

    Returns:
        pitch (ndarray): The predicted pitch.
        periodicity (ndarray): The predicted periodicity.
    """
    # torchcrepe inference
    pitch, periodicity = torchcrepe.predict(
        audio,
        fmin=50.0,
        fmax=550,
        sample_rate=torchcrepe.SAMPLE_RATE,
        model="full",
        return_periodicity=True,
        device=audio.device,
        pad=False,
    )
    pitch = pitch.cpu().numpy()
    periodicity = periodicity.cpu().numpy()

    # Calculate dB-scaled spectrogram and set low energy frames to unvoiced
    hop_length = torchcrepe.SAMPLE_RATE // 100  # default CREPE
    stft = torchaudio.functional.spectrogram(
        audio,
        window=torch.hann_window(torchcrepe.WINDOW_SIZE, device=audio.device),
        n_fft=torchcrepe.WINDOW_SIZE,
        hop_length=hop_length,
        win_length=torchcrepe.WINDOW_SIZE,
        power=2,
        normalized=False,
        pad=0,
        center=False,
    )

    # Perceptual weighting
    freqs = librosa.fft_frequencies(sr=torchcrepe.SAMPLE_RATE, n_fft=torchcrepe.WINDOW_SIZE)
    perceptual_stft = librosa.perceptual_weighting(stft.cpu().numpy(), freqs) - REF_DB
    silence = perceptual_stft.mean(axis=1) < silence_threshold

    periodicity[silence] = 0
    pitch[periodicity < unvoiced_treshold] = torchcrepe.UNVOICED

    return pitch, periodicity
