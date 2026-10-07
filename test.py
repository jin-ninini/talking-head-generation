"""Test a checkpoint by generating a lip-synced video (raw, or --postprocess with Poisson blending + GFPGAN)."""
import argparse

from src.inference import DEFAULT_AUDIO_DIR1, DEFAULT_AUDIO_DIR2, run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir",     required=True)
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--cfg",         required=True)
    parser.add_argument("--out",         default="data/output/inference/D1")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--audio_dir1",  default=DEFAULT_AUDIO_DIR1)
    parser.add_argument("--audio_dir2",  default=DEFAULT_AUDIO_DIR2)
    parser.add_argument("--test",        action="store_true", help="quick run: first window only")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--postprocess", action="store_true", help="Poisson blending + GFPGAN")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
