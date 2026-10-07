"""Preprocess a new input: `video` (face crops, env adlip-prep) or `audio` (wav + Whisper + mel, env thg)."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    v = sub.add_parser("video", help="frames/, face_frames/, lip_meta.npy, frame_indices.npy")
    v.add_argument("--video", required=True)
    v.add_argument("--out",   required=True)
    a = sub.add_parser("audio", help="audio.wav, audio_feature.npy, mel_latentsync.npy")
    a.add_argument("--src",    required=True, help="video or audio file")
    a.add_argument("--out",    required=True)
    a.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Imported per step: mediapipe (video) and transformers (audio) live in different envs.
    if args.step == "video":
        from src.preprocess_video import preprocess
        preprocess(Path(args.video), Path(args.out), print)
    else:
        from src.preprocess_audio import preprocess_audio
        af, mel = preprocess_audio(args.src, args.out, device=args.device)
        print(f"audio_feature {af}, mel_latentsync {mel}")


if __name__ == "__main__":
    main()
