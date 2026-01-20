import os
import json
import torch
import torchaudio
from model import MurmurModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "hs_murmur_model_v3.pt")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)


SAMPLE_RATE = CONFIG["sample_rate"]
MAX_LEN = SAMPLE_RATE * CONFIG["max_duration_sec"]
THRESHOLD = CONFIG["decision_threshold"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MurmurModel().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()


def preprocess_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    if wav.shape[0] < MAX_LEN:
        wav = torch.nn.functional.pad(
            wav, (0, MAX_LEN - wav.shape[0])
        )
    wav = wav / (wav.abs().max() + 1e-6)
    wav = wav[:MAX_LEN]
    return wav.unsqueeze(0)


def predict_murmur(wav_path: str) -> dict:

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    wav = preprocess_audio(wav_path).to(DEVICE)

    with torch.no_grad():
        logits = model(wav)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    pred = int(prob > THRESHOLD)

    return {
        "murmur_probability": round(prob, 4),
        "prediction": "Present" if pred == 1 else "Absent"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Heart Sound Murmur Inference"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to .wav file"
    )

    args = parser.parse_args()

    result = predict_murmur(args.audio)
    print(result)