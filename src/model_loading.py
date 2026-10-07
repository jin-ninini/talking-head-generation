import torch

from src.dataset import N_FRAMES
from src.vae_wrapper import VAEWrapper
from src.generator import TemporalLipGenerator
from src.latentsync import SyncNet as LatentSyncNet


def load_frozen_vae(model_name, device):
    """Frozen SD VAE + latent shape of a 96x96 lip image ([1, C, H, W])."""
    vae = VAEWrapper(model_name).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        z_dummy = vae.encode(torch.zeros(1, 3, 96, 96).to(device))
    return vae, z_dummy.shape


def build_generator(cfg, latent_shape, device):
    _, C, H, W = latent_shape
    return TemporalLipGenerator(
        latent_dim=C, spatial_len=H*W, audio_in=384,
        audio_len=cfg.model.audio_len, n_frames=N_FRAMES,
        dim=cfg.model.dim, num_heads=cfg.model.num_heads,
        spatial_layers=cfg.model.spatial_layers,
        temporal_layers=cfg.model.temporal_layers,
        ffn_dim_ratio=cfg.model.ffn_dim_ratio,
        dropout=cfg.model.dropout, alpha=cfg.model.alpha,
    ).to(device)


# ──────────────────────────────────────────
# SyncNet
# ──────────────────────────────────────────
SYNCNET_CONFIG = {
    "audio_encoder": {
        "in_channels": 1,
        "block_out_channels": [32,64,128,256,512,1024,2048],
        "downsample_factors": [[2,1],2,2,1,2,2,[2,3]],
        "attn_blocks": [0,0,0,0,0,0,0],
        "dropout": 0.0,
    },
    "visual_encoder": {
        "in_channels": 48,
        "block_out_channels": [64,128,256,256,512,1024,2048,2048],
        "downsample_factors": [[1,2],2,2,2,2,2,2,2],
        "attn_blocks": [0,0,0,0,0,0,0,0],
        "dropout": 0.0,
    },
}

def load_latentsync(ckpt_path, device):
    model = LatentSyncNet(SYNCNET_CONFIG).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"LatentSync loaded & frozen: {ckpt_path}", flush=True)
    return model
