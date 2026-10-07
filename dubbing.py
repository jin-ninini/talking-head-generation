"""Dub English speech into Korean: Wav2Vec2 STT → mBART-50 translation → FastSpeech2 + VocGAN TTS (env dub)."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="English speech (any sample rate)")
    parser.add_argument("--out",   required=True, help="Output dir: korean.wav, transcript.tsv, segments/")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Imported here: loading the module downloads and loads the STT / MT models.
    from src.dubbing.speech_translation_pipeline import run_pipeline
    run_pipeline(os.path.abspath(args.input), os.path.abspath(args.out))


if __name__ == "__main__":
    main()
