"""Validate a checkpoint: latent L1 on the val split (same metric as during training)."""
import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import LipSyncDataset
from src.inference import load_model
from src.seed import set_seed
from src.validation import validate_latent_l1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",    required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.cfg)
    device = torch.device(args.device)
    set_seed(cfg.train.seed)

    vae, model, *_ = load_model(cfg, args.ckpt, device)

    val_dataset = LipSyncDataset(
        cfg.data.root, split="val",
        samples_per_video=cfg.data.samples_per_video,
        splits_dir=cfg.data.get("splits_dir", None),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    val_loss = validate_latent_l1(model, vae, val_loader, device,
                                  ref_scale=cfg.model.get("ref_scale", 1.0))
    print(f"[val] loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
