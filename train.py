"""Train the ADLip2 generator (single GPU by default, --ddp for multi-GPU)."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",     required=True)
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--ckpt",    default=None)
    parser.add_argument("--ddp",     action="store_true", help="multi-GPU training (launch with torchrun)")
    parser.add_argument("--out_dir", default=None, help="--ddp only: fixed output directory")
    args = parser.parse_args()

    # Imported after parsing: each trainer sets CUDA_VISIBLE_DEVICES before torch is imported.
    if args.ddp:
        from src.trainer_ddp import train
        train(args.cfg, args.device, args.ckpt, args.out_dir)
    else:
        from src.trainer import train
        train(args.cfg, args.device, args.ckpt)


if __name__ == "__main__":
    main()
