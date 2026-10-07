"""LSE-C / LSE-D of generated videos with syncnet_python (env dub, ffmpeg on PATH)."""
import argparse
import csv

from src.lse import lse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True, help="name=path.mp4 pairs")
    parser.add_argument("--syncnet_dir", required=True, help="syncnet_python clone with downloaded models")
    parser.add_argument("--work_dir", required=True, help="scratch dir for face tracks")
    parser.add_argument("--out", required=True, help="output csv")
    args = parser.parse_args()

    rows = []
    for item in args.videos:
        name, path = item.split("=", 1)
        c, d, n = lse(path, name, args.syncnet_dir, args.work_dir)
        rows.append([name, path, f"{c:.3f}", f"{d:.3f}", n])
        print(f"{name}: LSE-C {c:.3f}  LSE-D {d:.3f}  ({n} tracks)")

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([["video", "path", "lse_c", "lse_d", "tracks"]] + rows)


if __name__ == "__main__":
    main()
