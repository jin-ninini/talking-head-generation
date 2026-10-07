import numpy as np
import torch
import torch.nn.functional as F


def encode_seq(vae, lips):
    """[B, T, 3, 96, 96] → [B, T, C, H, W]"""
    b, t, c, h, w = lips.shape
    z = vae.encode(lips.reshape(b*t, c, h, w))
    return z.reshape(b, t, *z.shape[1:])


def validate_latent_l1(model, vae, loader, device, ref_scale=1.0):
    """Mean latent L1 between prediction (ref A + GT audio) and GT over the loader."""
    val_losses = []
    with torch.no_grad():
        for val_batch in loader:
            vt = val_batch["target_lips"].to(device)
            vA = val_batch["ref_lips_A"].to(device)
            va = val_batch["audios"].to(device)

            z_vA  = encode_seq(vae, vA) * ref_scale
            z_vgt = encode_seq(vae, vt)

            z_vo, _ = model(z_vA, va)
            val_losses.append(F.l1_loss(z_vo, z_vgt).item())
    return float(np.mean(val_losses))
