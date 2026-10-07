"""
src/inference.py - ADLip2 inference

ref 영상의 고정된 한 프레임(lip latent)과 audio로 립 영상을 생성해 얼굴에 붙인다.
  postprocess=False : feather mask paste           → simple_ref{N}.mp4
  postprocess=True  : Poisson blending + GFPGAN    → out_simple.mp4
"""

import logging, random, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.dataset import N_FRAMES, get_audio_windows
from src.model_loading import load_frozen_vae, build_generator

VIDEO_FPS    = 25
LIP_SIZE     = 96
FACE_SIZE    = 256
ENCODE_BATCH = 64

HDTF_ROOT          = Path("data/HDTF_processed")
DEFAULT_AUDIO_DIR1 = str(HDTF_ROOT / "WRA_ShelleyMooreCapito0_000")
DEFAULT_AUDIO_DIR2 = str(HDTF_ROOT / "RD_Radio5_000")


# ── helpers ───────────────────────────────────────────────────────
def read_img(path, size):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.tensor(cv2.resize(img, (size, size))).permute(2,0,1).float() / 127.5 - 1.0

def decode_seq(vae, z):
    b, t, c, h, w = z.shape
    imgs = vae.decode(z.reshape(b*t, c, h, w))
    return imgs.reshape(b, t, *imgs.shape[1:])

def make_feather_mask(h, w, feather=0.15, device="cpu"):
    ys = torch.linspace(0, 1, h, device=device)
    xs = torch.linspace(0, 1, w, device=device)
    fy = (ys.clamp(0, feather) / feather) * (1 - (ys - (1 - feather)).clamp(0, feather) / feather)
    fx = (xs.clamp(0, feather) / feather) * (1 - (xs - (1 - feather)).clamp(0, feather) / feather)
    mask = (fy.unsqueeze(1) * fx.unsqueeze(0)).unsqueeze(0)
    return mask.clamp(0, 1)

def paste_to_full_face(face_frs, pred_lips, lip_meta):
    result = face_frs.clone()
    for f in range(N_FRAMES):
        valid, _, x1, y1, x2, y2 = lip_meta[f].tolist()
        if valid != 1: continue
        x1i, x2i = max(0, min(255, int(x1))), max(1, min(256, int(x2)))
        y1i, y2i = max(0, min(255, int(y1))), max(1, min(256, int(y2)))
        if y2i <= y1i or x2i <= x1i: continue
        h_box, w_box = y2i - y1i, x2i - x1i
        resized = F.interpolate(pred_lips[f:f+1], size=(h_box, w_box),
                                mode="bilinear", align_corners=False)[0]
        mask = make_feather_mask(h_box, w_box, feather=0.15, device=face_frs.device)
        result[f, :, y1i:y2i, x1i:x2i] = (
            mask * resized + (1 - mask) * face_frs[f, :, y1i:y2i, x1i:x2i]
        )
    return result

def paste_to_full_face_poisson(face_frs, pred_lips, lip_meta):
    result = face_frs.clone()
    for f in range(N_FRAMES):
        valid, _, x1, y1, x2, y2 = lip_meta[f].tolist()
        if valid != 1: continue
        x1i, x2i = max(0, min(255, int(x1))), max(1, min(256, int(x2)))
        y1i, y2i = max(0, min(255, int(y1))), max(1, min(256, int(y2)))
        if y2i <= y1i or x2i <= x1i: continue
        h_box, w_box = y2i - y1i, x2i - x1i

        face_np = to_uint8(face_frs[f])  # HxWx3
        resized = F.interpolate(pred_lips[f:f+1], size=(h_box, w_box),
                                mode="bilinear", align_corners=False)[0]
        lip_np = to_uint8(resized)  # h_box x w_box x 3

        # Poisson blending
        mask = 255 * np.ones((h_box, w_box), dtype=np.uint8)
        center = (x1i + w_box // 2, y1i + h_box // 2)
        try:
            blended = cv2.seamlessClone(
                cv2.cvtColor(lip_np, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR),
                mask, center, cv2.NORMAL_CLONE
            )
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            result[f] = torch.tensor(blended_rgb).permute(2,0,1).float() / 127.5 - 1.0
        except:
            # 실패하면 기존 feather mask로 fallback
            mask_t = make_feather_mask(h_box, w_box, feather=0.15, device=face_frs.device)
            result[f, :, y1i:y2i, x1i:x2i] = (
                mask_t * resized + (1 - mask_t) * face_frs[f, :, y1i:y2i, x1i:x2i]
            )
    return result

def to_uint8(t):
    return ((t.clamp(-1,1)+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

def enhance_with_gfpgan(gfpgan, img_rgb):
    _, _, output = gfpgan.enhance(
        img_rgb,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5,
    )
    if output is None:
        return img_rgb

    # 선명도(Y)만 GFPGAN, 색상(Cb, Cr)은 원본 유지
    orig_ycbcr     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    enhanced_ycbcr = cv2.cvtColor(output,  cv2.COLOR_RGB2YCrCb)
    enhanced_ycbcr[:, :, 1] = orig_ycbcr[:, :, 1]  # Cr 원본
    enhanced_ycbcr[:, :, 2] = orig_ycbcr[:, :, 2]  # Cb 원본
    return cv2.cvtColor(enhanced_ycbcr, cv2.COLOR_YCrCb2RGB)

def frames_to_combined_mp4(lip_frames, face_frames, path, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (FACE_SIZE*2, FACE_SIZE))
    for lip, face in zip(lip_frames, face_frames):
        lip_up = cv2.resize(lip, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        vw.write(cv2.cvtColor(np.concatenate([lip_up, face], axis=1), cv2.COLOR_RGB2BGR))
    vw.release()

def mux(silent, wav, out):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(silent), "-i", str(wav),
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except:
        return False


# ── 모델 로드 ─────────────────────────────────────────────────────
def load_model(cfg, ckpt_path, device):
    vae, latent_shape = load_frozen_vae(cfg.vae.model_name, device)
    _, C, H, W = latent_shape

    model = build_generator(cfg, latent_shape, device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd  = raw["model"] if "model" in raw else raw
    model.load_state_dict(sd, strict=False)
    model.eval()
    return vae, model, C, H, W

def load_gfpgan(device):
    from gfpgan import GFPGANer
    return GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


# ── ref VAE 캐시 ──────────────────────────────────────────────────
def cache_z_ref(vae, lip_paths, n_ref, device):
    all_z = []
    with torch.no_grad():
        for i in range(0, n_ref, ENCODE_BATCH):
            batch = [read_img(lip_paths[j], LIP_SIZE)
                     for j in range(i, min(i+ENCODE_BATCH, n_ref))]
            z = vae.encode(torch.stack(batch).to(device))
            all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


# ── 단일 윈도우 inference ─────────────────────────────────────────
def infer_window(model, vae, z_ref_t, audios, face_paths, lip_meta_all,
                 t_ref, n_ref, device, poisson=False):
    with torch.no_grad():
        z_out, _ = model(z_ref_t, audios)
        pred     = decode_seq(vae, z_out)[0]   # [16, 3, 96, 96]

    face_imgs = []
    ref_face = read_img(face_paths[t_ref], FACE_SIZE)
    face_imgs = [ref_face] * N_FRAMES

    face_frs = torch.stack(face_imgs).to(device)
    meta = np.stack([lip_meta_all[t_ref]] * N_FRAMES)
    if poisson:
        pasted = paste_to_full_face_poisson(face_frs, pred, meta)
    else:
        pasted = paste_to_full_face(face_frs, pred, meta)

    lips  = [to_uint8(pred[f])   for f in range(N_FRAMES)]
    faces = [to_uint8(pasted[f]) for f in range(N_FRAMES)]
    return lips, faces


# ── 실험 실행 ─────────────────────────────────────────────────────
def run_experiment(label, model, vae, all_z_ref,
                   face_paths, lip_meta_all, n_ref,
                   audio_feature, wav_path,
                   ref_start, out_dir, device, info, suffix,
                   max_windows=None, gfpgan=None, poisson=False):
    n_audio   = len(audio_feature)
    n_windows = (n_audio - N_FRAMES) // N_FRAMES + 1

    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    all_lips, all_faces = [], []
    z_single = all_z_ref[ref_start:ref_start+1]          # [1, C, H, W]
    z_ref_fixed = z_single.repeat(N_FRAMES, 1, 1, 1).unsqueeze(0).to(device)  # [1, 16, C, H, W]

    for win_idx in range(n_windows):
        t_audio    = win_idx * N_FRAMES
        t_ref_face = ref_start

        audios = get_audio_windows(audio_feature, t_audio, n_audio).unsqueeze(0).to(device)
        lips, faces = infer_window(model, vae, z_ref_fixed, audios,
                                   face_paths, lip_meta_all, t_ref_face, n_ref, device,
                                   poisson=poisson)
        all_lips.extend(lips)
        all_faces.extend(faces)

    if gfpgan is not None:
        all_faces = [enhance_with_gfpgan(gfpgan, f) for f in all_faces]

    out_name = f"{label}_{suffix}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "s.mp4"
        frames_to_combined_mp4(all_lips, all_faces, silent)
        final = out_dir / out_name
        if wav_path and Path(wav_path).exists():
            ok = mux(silent, wav_path, final)
            if not ok:
                shutil.copy(silent, final)
        else:
            shutil.copy(silent, final)
    info(f"saved: {out_name}")
    return final


# ── 실행 ─────────────────────────────────────────────────────────
def run_inference(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    device     = torch.device(args.device)
    ref_dir    = Path(args.ref_dir)
    out_dir    = Path(args.out)
    audio_dir1 = Path(args.audio_dir1)
    audio_dir2 = Path(args.audio_dir2)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(out_dir / "eval.log"), level=logging.INFO,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True,
    )
    log = logging.getLogger()
    def info(msg): print(msg, flush=True); log.info(msg)

    lip_paths  = sorted((ref_dir / "frames").glob("*.png"))
    face_paths = sorted((ref_dir / "face_frames").glob("*.png"))
    lip_meta   = np.load(str(ref_dir / "lip_meta.npy"))
    n_ref      = min(len(lip_paths), len(face_paths), len(lip_meta))
    info(f"ref: {ref_dir.name} | 프레임: {n_ref}")

    audio1_af  = np.load(str(audio_dir1 / "audio_feature.npy"))
    audio1_wav = audio_dir1 / "audio.wav"
    audio2_af  = np.load(str(audio_dir2 / "audio_feature.npy"))
    audio2_wav = audio_dir2 / "audio.wav"
    info(f"audio1: {audio_dir1.name}")
    info(f"audio2: {audio_dir2.name}")

    # A1/A2는 현재 사용되지 않지만, ref_start_B 난수 순서를 유지하기 위해 그대로 둔다
    max_start    = max(1, n_ref - N_FRAMES)
    ref_start_A1 = random.randint(0, max_start - 1)
    ref_start_A2 = random.randint(0, max_start - 1)
    while abs(ref_start_A2 - ref_start_A1) < 50:
        ref_start_A2 = random.randint(0, max_start - 1)
    ref_start_B = random.randint(0, max_start - 1)

    info(f"[ref 인덱스] ExpA ref1={ref_start_A1} ref2={ref_start_A2} ExpB/C={ref_start_B}")

    cfg = OmegaConf.load(args.cfg)
    vae, model, C, H, W = load_model(cfg, args.ckpt, device)
    info(f"모델 로드 완료: {args.ckpt}")
    gfpgan = None
    if args.postprocess:
        gfpgan = load_gfpgan(device)
        info("GFPGAN 로드 완료")

    info("ref VAE encode 중...")
    all_z_ref = cache_z_ref(vae, lip_paths, n_ref, device)
    info(f"z_ref: {all_z_ref.shape}")

    mw = 1 if args.test else args.max_windows

    info(f"\n[Simple] ref={ref_dir.name} audio={audio_dir1.name}")
    if args.postprocess:
        label, suffix = "out", "simple"
    else:
        label, suffix = "simple", f"ref{ref_start_B}"
    run_experiment(label, model, vae, all_z_ref,
                   face_paths, lip_meta, n_ref,
                   audio1_af, audio1_wav, ref_start_B,
                   out_dir, device, info, suffix, max_windows=mw,
                   gfpgan=gfpgan, poisson=args.postprocess)

    info(f"\n완료: {out_dir}")
