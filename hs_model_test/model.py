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

        self.attn_V = nn.Linear(hidden, hidden)
        self.attn_U = nn.Linear(hidden, hidden)
        self.attn_w = nn.Linear(hidden, 1)


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

        frame_mask = (mel.sum(dim=1) != 0)

        MAX_FRAMES = 3000
        T = mel.shape[-1]

        if T > MAX_FRAMES:
            mel = mel[..., :MAX_FRAMES]
            frame_mask = frame_mask[:, :MAX_FRAMES]
        else:
            pad_len = MAX_FRAMES - T
            mel = torch.nn.functional.pad(mel, (0, pad_len))
            frame_mask = torch.nn.functional.pad(
                frame_mask, (0, pad_len), value=False
            )

        feats = self.encoder(input_features=mel).last_hidden_state
        frame_mask = frame_mask[:, ::2][:, :feats.shape[1]]
        feats = self.temporal(
            feats,
            src_key_padding_mask=~frame_mask
        )

        A_V = torch.tanh(self.attn_V(feats))
        A_U = torch.sigmoid(self.attn_U(feats))
        A = self.attn_w(A_V * A_U)

        weights = torch.softmax(A, dim=1)
        pooled = (weights * feats).sum(dim=1)


        return self.classifier(pooled)
