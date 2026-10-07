import torch
import torch.nn.functional as F

from src.dataset import N_FRAMES

# ──────────────────────────────────────────
# paste pred_lips → face_window
# ──────────────────────────────────────────
def paste_lips_to_face(face_window, pred_lips, lip_bbox):
    """
    face_window: [B, 48, 128, 256]  # 하단 절반
    pred_lips:   [B, 16, 3, 96, 96]
    lip_bbox:    [B, 16, 6] = [valid, frame_idx, x1, y1, x2, y2] (원본 256x256 기준)
    """
    B = face_window.shape[0]
    result = face_window.clone()

    for b in range(B):
        for f in range(N_FRAMES):
            valid, fidx, x1, y1, x2, y2 = lip_bbox[b, f].tolist()
            if valid != 1:
                continue
            # 원본 256×256 → 하단절반 128×256 좌표 변환
            y1h = max(0,   min(127, int(y1) - 128))
            y2h = max(1,   min(128, int(y2) - 128))
            x1i = max(0,   min(255, int(x1)))
            x2i = max(1,   min(256, int(x2)))
            if y2h <= y1h or x2i <= x1i:
                continue
            ph = y2h - y1h
            pw = x2i - x1i
            lip_resized = F.interpolate(
                pred_lips[b, f:f+1],
                size=(ph, pw),
                mode='bilinear',
                align_corners=False,
            )[0]
            ch_start = f * 3
            result[b, ch_start:ch_start+3, y1h:y2h, x1i:x2i] = lip_resized

    return result


def crop_mouth(x):
    if x.dim() == 5:
        return x[:, :, :, 48:96, 16:80]
    elif x.dim() == 4:
        return x[:, :, 48:96, 16:80]
    else:
        raise ValueError(f"Unexpected shape: {x.shape}")

def sobel_edge(x):
    """x: [B, 3, H, W] → edge map [B, 1, H, W]"""
    x = x.float()  # fp16 → fp32 강제
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       dtype=torch.float32, device=x.device).reshape(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                       dtype=torch.float32, device=x.device).reshape(1,1,3,3)
    ex = F.conv2d(gray, kx, padding=1)
    ey = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(ex**2 + ey**2 + 1e-6)

def sync_loss_contrastive(face_embed, audio_embed, margin=0.2):
    pos  = F.cosine_similarity(face_embed, audio_embed, dim=1)
    perm = torch.roll(torch.arange(face_embed.shape[0], device=face_embed.device), 1)
    neg  = F.cosine_similarity(face_embed, audio_embed[perm], dim=1)
    loss = F.relu(margin - pos + neg).mean()
    return loss, pos.mean().item(), neg.mean().item()
