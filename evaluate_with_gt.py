import os
import numpy as np
import torch
import torchaudio
import torchcrepe
import soundfile as sf
from typing import Dict
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from python_speech_features import mfcc
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score

# ----------------------------
# Compute Metrics
# ----------------------------

# MCD
def compute_mcd(ref, deg, sr):
    ref_mfcc = mfcc(ref, sr)
    deg_mfcc = mfcc(deg, sr)
    length = min(len(ref_mfcc), len(deg_mfcc))
    return np.mean([euclidean(r, d) for r, d in zip(ref_mfcc[:length], deg_mfcc[:length])])

# SNR
def compute_snr(ref, deg):
    noise = ref - deg
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum(noise ** 2) + 1e-8))

# LSD
def compute_lsd(ref, deg):
    spec_ref = np.abs(np.fft.rfft(ref))
    spec_deg = np.abs(np.fft.rfft(deg))
    return np.mean(np.sqrt((20 * np.log10((spec_ref + 1e-8)/(spec_deg + 1e-8)))**2))

# Pitch RMSE
def compute_f0_rmse(ref_audio, gen_audio, sr):
    ref_tensor = torch.tensor(ref_audio).unsqueeze(0)
    gen_tensor = torch.tensor(gen_audio).unsqueeze(0)
    pitch_ref, _ = torchaudio.functional.detect_pitch_frequency(ref_tensor, sr)
    pitch_gen, _ = torchaudio.functional.detect_pitch_frequency(gen_tensor, sr)
    return torch.sqrt(torch.mean((pitch_ref - pitch_gen) ** 2)).item()


# Periodicity_F1 
# vocos 에서 소스 가져오기..
def compute_periodicity_f1(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int) -> float:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # 길이 맞추기
    # min_len = min(len(ref_audio), len(gen_audio))
    # ref_audio = ref_audio[:min_len]
    # gen_audio = gen_audio[:min_len]

    # # torchcrepe가 16000Hz를 권장하므로 resample 필요할 수도 있음
    # if sr != 16000:
    #     import torchaudio
    #     ref_audio = torchaudio.functional.resample(torch.tensor(ref_audio), sr, 16000).numpy()
    #     gen_audio = torchaudio.functional.resample(torch.tensor(gen_audio), sr, 16000).numpy()
    #     sr = 16000

    # # torch tensor로 변환
    # ref_tensor = torch.tensor(ref_audio, dtype=torch.float32).to(device)
    # gen_tensor = torch.tensor(gen_audio, dtype=torch.float32).to(device)

    # # torchcrepe expects (batch, time), batch size 1
    # ref_tensor = ref_tensor.unsqueeze(0)
    # gen_tensor = gen_tensor.unsqueeze(0)

    # # pitch estimation (Hz), confidence (0~1)
    # ref_pitch, ref_confidence = torchcrepe.predict(ref_tensor, sr, step_size=10, batch_size=512, device=device, float32=True)
    # gen_pitch, gen_confidence = torchcrepe.predict(gen_tensor, sr, step_size=10, batch_size=512, device=device, float32=True)

    # # voiced/unvoiced 판정: confidence > 0.5, pitch > 0 => voiced (1), 아니면 unvoiced (0)
    # ref_voiced = ((ref_confidence > 0.5) & (ref_pitch > 0)).squeeze().cpu().numpy().astype(int)
    # gen_voiced = ((gen_confidence > 0.5) & (gen_pitch > 0)).squeeze().cpu().numpy().astype(int)

    # # 길이 맞추기
    # min_frames = min(len(ref_voiced), len(gen_voiced))
    # ref_voiced = ref_voiced[:min_frames]
    # gen_voiced = gen_voiced[:min_frames]

    # # F1 score 계산
    # f1 = f1_score(ref_voiced, gen_voiced)

    return f1

