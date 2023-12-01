import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

import glob
import os
import numpy as np
import time
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')
spk_encoder.load_model(enc_model_fpath, device="cuda")
for dir in os.listdir('./data/wavs'):
    print(dir)
    os.makedirs('./data/mels/'+dir,exist_ok=True)
    os.makedirs('./data/embeds/'+dir,exist_ok=True)
    for wav in os.listdir('./data/wavs/'+dir):
       spec = get_mel('./data/wavs/'+dir+'/'+wav)
       np.save('./data/mels/'+dir+'/'+wav.replace('.wav','_mel.npy'),spec)
       emb = get_embed('./data/wavs/'+dir+'/'+wav)
       np.save('./data/embeds/'+dir+'/'+wav.replace('.wav','_embed.npy'),emb)
       time.sleep(1)

