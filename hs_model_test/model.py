import os
import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel
import json
torchaudio.set_audio_backend("soundfile")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

SAMPLE_RATE = CONFIG["sample_rate"]
MAX_LEN = SAMPLE_RATE * CONFIG["max_duration_sec"]
THRESHOLD = CONFIG["decision_threshold"]

class MurmurModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=128,
            hop_length=32,
            n_mels=80,
            f_min=20,
            f_max=800
        )

        self.encoder = WhisperModel.from_pretrained(
            "openai/whisper-small"
        ).encoder

        for p in self.encoder.parameters():
            p.requires_grad = False
            
        hidden = self.encoder.config.hidden_size

        self.attn = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softmax(dim=1)
        )
        
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, wav):
        mel = torch.log(self.mel(wav) + 1e-6)

        MAX_FRAMES = 3000
        T = mel.shape[-1]

        if T > MAX_FRAMES:
            mel = mel[..., :MAX_FRAMES]

        else:
            mel = torch.nn.functional.pad(
                mel, (0, MAX_FRAMES - T)
            )

        feats = self.encoder(input_features=mel).last_hidden_state
        feats = self.temporal(feats)

        weights = self.attn(feats)
        pooled = (weights * feats).sum(dim=1)
        return self.classifier(pooled)