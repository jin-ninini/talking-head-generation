"""Audio preprocessing: 16 kHz wav, per-frame Whisper features, LatentSync mel.

extract_audio() and WhisperFeatureExtractor are copied verbatim from ADLip `data/preprocess.py`
(https://github.com/Inha-Jeong/ADLip). The mel_latentsync.npy producer was not in any team repo;
it is rebuilt with LatentSync's melspectrogram (src/latentsync_audio.py), the format the trainer
feeds to the LatentSync SyncNet (80 mels, 80 frames/s).
"""
import math
import subprocess
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WhisperModel

from src import latentsync_audio


def extract_audio(video_path, out_path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1",
        "-vn", str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class WhisperFeatureExtractor:
    def __init__(self, model_name="openai/whisper-tiny", device="cuda"):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, wav_path, fps=25):
        audio, sr = librosa.load(wav_path, sr=16000)
        assert sr == 16000

        # 30초 단위로 분할 (Whisper 입력 제한)
        segment_length = 30 * sr
        segments = [audio[i:i + segment_length]
                    for i in range(0, len(audio), segment_length)]

        all_features = []
        for seg in segments:
            inp = self.feature_extractor(
                seg, return_tensors="pt", sampling_rate=sr
            ).input_features.to(self.device)
            feat = self.model.encoder(inp).last_hidden_state  # [1, T, 384]
            all_features.append(feat)

        whisper_feature = torch.cat(all_features, dim=1)  # [1, T_total, 384]

        # 실제 오디오 길이에 맞게 trim
        audio_fps = 50
        actual_length = math.floor((len(audio) / sr) * audio_fps)
        whisper_feature = whisper_feature[:, :actual_length, :]  # [1, T, 384]

        # 프레임별 feature 슬라이싱
        # whisper 50fps, 영상 25fps → 프레임 i = whisper token i*2
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((len(audio) / sr) * fps)

        frame_features = []
        for i in range(num_frames):
            idx = min(
                math.floor(i * whisper_idx_multiplier),
                whisper_feature.shape[1] - 1
            )
            frame_features.append(whisper_feature[0, idx, :].cpu().numpy())

        return np.array(frame_features)  # [num_frames, 384]


def preprocess_audio(src_path, out_dir, device="cuda"):
    """src_path (video or audio file) → out_dir/{audio.wav, audio_feature.npy, mel_latentsync.npy}"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio.wav"
    extract_audio(src_path, wav_path)

    audio_feature = WhisperFeatureExtractor(device=device).extract(str(wav_path), fps=25)
    np.save(str(out_dir / "audio_feature.npy"), audio_feature)

    wav = latentsync_audio.load_wav(str(wav_path), 16000)
    mel = latentsync_audio.melspectrogram(wav).astype(np.float32)
    np.save(str(out_dir / "mel_latentsync.npy"), mel)
    return audio_feature.shape, mel.shape
