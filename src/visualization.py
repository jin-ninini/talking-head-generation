from pathlib import Path

import cv2
import numpy as np
import torch

from src.dataset import N_FRAMES, get_audio_windows

CENTER = N_FRAMES // 2  # 8

# ──────────────────────────────────────────
# 샘플 저장
# ──────────────────────────────────────────
def save_samples(model, vae, cfg, device, samples_dir, epoch, writer=None):
    """
    center frame 비교 이미지 + sequence 이미지 저장
    GT/Ref/Pred 모두 CENTER frame으로 통일
    """
    def load_lip_window(frame_dir, start, n_total):
        imgs = []
        for i in range(N_FRAMES):
            idx = min(start + i, n_total - 1)
            path = frame_dir / f"{idx:05d}.png"
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))
            t = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(device) / 127.5 - 1.0
            imgs.append(t)
        return torch.cat(imgs, dim=0).unsqueeze(0)  # [1, 16, 3, 96, 96]

    def to_np(t):
        return ((t.clamp(-1,1)+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

    def put_label(img, text):
        img = img.copy()
        cv2.putText(img, text, (3,12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        return img

    samples = [
        ("WDA_HillaryClinton_000", 68,  648, 826, "Clinton"),
        ("RD_Radio5_000",          660, 6,   608, "Radio5"),
        ("RD_Radio27_000",         1,   13,  131, "Radio27"),
    ]
    # Fixed HDTF clips; skipped when data.root does not contain them (e.g. configs/demo.yaml)
    samples = [s for s in samples if (Path(cfg.data.root) / s[0]).exists()]
    if not samples:
        return

    center_rows = []
    seq_rows    = []

    with torch.no_grad():
        for video_name, t, j_A, j_B, label in samples:
            base = Path(cfg.data.root) / video_name
            af   = np.load(str(base / "audio_feature.npy"))
            n    = len(af)
            fd   = base / "frames"

            target_lips = load_lip_window(fd, t,   n)  # [1,16,3,96,96]
            ref_lips_A  = load_lip_window(fd, j_A, n)
            ref_lips_B  = load_lip_window(fd, j_B, n)
            audios      = get_audio_windows(af, t, n).unsqueeze(0).to(device)   # [1,16,5,384]
            zero_audio  = torch.zeros_like(audios)

            # neg audio: center frame 기준 100프레임 뒤
            neg_start = min(t + 100, n - N_FRAMES)
            audio_neg = get_audio_windows(af, neg_start, n).unsqueeze(0).to(device)

            z_A  = vae.encode(ref_lips_A.reshape(-1, 3, 96, 96))
            _, _, H, W = z_A.shape
            z_A  = z_A.reshape(1, N_FRAMES, -1, H, W)
            z_B  = vae.encode(ref_lips_B.reshape(-1, 3, 96, 96)).reshape(1, N_FRAMES, -1, H, W)

            z_out_A,    _ = model(z_A, audios)
            z_out_B,    _ = model(z_B, audios)
            z_out_zero, _ = model(z_A, zero_audio)
            z_out_neg,  _ = model(z_A, audio_neg)

            def decode_seq(z):
                imgs = vae.decode(z.reshape(-1, z.shape[2], H, W))
                return imgs.reshape(1, N_FRAMES, 3, 96, 96)

            pred_A    = decode_seq(z_out_A)
            pred_B    = decode_seq(z_out_B)
            pred_zero = decode_seq(z_out_zero)
            pred_neg  = decode_seq(z_out_neg)

            # GT/Ref/Pred 모두 CENTER frame으로 통일
            c = CENTER
            gt_c  = to_np(target_lips[0, c])
            rA_c  = to_np(ref_lips_A[0, c])
            rB_c  = to_np(ref_lips_B[0, c])
            pA_c  = to_np(pred_A[0, c])
            pB_c  = to_np(pred_B[0, c])
            pZ_c  = to_np(pred_zero[0, c])
            pN_c  = to_np(pred_neg[0, c])

            center_row = np.concatenate([
                put_label(gt_c,  f"GT(f{t+c})"),
                put_label(rA_c,  f"RefA(f{j_A+c})"),
                put_label(rB_c,  f"RefB(f{j_B+c})"),
                put_label(pA_c,  "PredA"),
                put_label(pB_c,  "PredB"),
                put_label(pZ_c,  "PredZero"),
                put_label(pN_c,  "PredNeg"),
            ], axis=1)
            center_rows.append(center_row)

            # sequence: 각 샘플별 7행 block
            def label_first(imgs, text):
                imgs = [x.copy() for x in imgs]
                cv2.putText(imgs[0], text, (3,12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                return imgs

            gt_seq    = [to_np(target_lips[0, f]) for f in range(N_FRAMES)]
            refA_seq  = [to_np(ref_lips_A[0, f])  for f in range(N_FRAMES)]
            refB_seq  = [to_np(ref_lips_B[0, f])  for f in range(N_FRAMES)]
            predA_seq = [to_np(pred_A[0, f])      for f in range(N_FRAMES)]
            predB_seq = [to_np(pred_B[0, f])      for f in range(N_FRAMES)]
            zero_seq  = [to_np(pred_zero[0, f])   for f in range(N_FRAMES)]
            neg_seq   = [to_np(pred_neg[0, f])    for f in range(N_FRAMES)]

            block = np.concatenate([
                np.concatenate(label_first(gt_seq,    f"GT   f{t}~{t+15}"),    axis=1),
                np.concatenate(label_first(refA_seq,  f"RefA f{j_A}~{j_A+15}"), axis=1),
                np.concatenate(label_first(refB_seq,  f"RefB f{j_B}~{j_B+15}"), axis=1),
                np.concatenate(label_first(predA_seq, "PredA (RefA+GT audio)"),  axis=1),
                np.concatenate(label_first(predB_seq, "PredB (RefB+GT audio)"),  axis=1),
                np.concatenate(label_first(zero_seq,  "PredZero (RefA+zero)"),   axis=1),
                np.concatenate(label_first(neg_seq,   "PredNeg (RefA+neg)"),     axis=1),
            ], axis=0)
            seq_rows.append(block)

    center_combined = cv2.cvtColor(np.concatenate(center_rows, axis=0), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(samples_dir / f"center_epoch_{epoch:04d}.png"), center_combined)

    sample_names = ["Clinton", "Radio5", "Radio27"]
    for block, name in zip(seq_rows, sample_names):
        img_bgr = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(samples_dir / f"seq_{name}_epoch_{epoch:04d}.png"), img_bgr)

    print(f"  samples saved: epoch {epoch}", flush=True)

    if writer is not None:
        img_t = torch.tensor(cv2.cvtColor(center_combined, cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255.0
        writer.add_image("sample/center", img_t, epoch)
        for block, name in zip(seq_rows, sample_names):
            img_t = torch.tensor(block).permute(2,0,1).float() / 255.0
            writer.add_image(f"sample/seq_{name}", img_t, epoch)
