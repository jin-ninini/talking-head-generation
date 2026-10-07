"""Video preprocessing: face/lip crops for one video.

Copied verbatim from ADLip `scripts/inference_raw_video.py` (constants + preprocess()),
https://github.com/Inha-Jeong/ADLip. Needs mediapipe with `mp.solutions` (env adlip-prep).
"""
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

N_FRAMES   = 16
VIDEO_FPS  = 25
LIP_SIZE   = 96
FACE_SIZE  = 256
AUDIO_HALF = 2
PAD        = 10

LIP_LANDMARKS = [
    61,185,40,39,37,0,267,269,270,409,
    291,375,321,405,314,17,84,181,91,146
]
LEFT_EYE  = 33
RIGHT_EYE = 263


# ── 전처리 ────────────────────────────────────────────────────────
def preprocess(video_path: Path, out_dir: Path, info):
    """
    frames/       ← lip crop 96×96 (affine aligned)
    face_frames/  ← full face 256×256 (affine aligned)
    lip_meta.npy  ← [N, 6] = [valid, frame_idx, x1, y1, x2, y2] (256×256 기준)
    frame_indices.npy
    """
    frames_dir = out_dir / "frames"
    face_dir   = out_dir / "face_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    face_mesh_video = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )
    face_mesh_static = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    cap       = cv2.VideoCapture(str(video_path))
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info(f"영상: {total}프레임")

    frame_idx     = 0
    saved_indices = []
    lip_meta_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh_video.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # affine alignment
            lx = int(lm[LEFT_EYE].x * w);  ly = int(lm[LEFT_EYE].y * h)
            rx = int(lm[RIGHT_EYE].x * w); ry = int(lm[RIGHT_EYE].y * h)
            angle = math.degrees(math.atan2(ry - ly, rx - lx))
            M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            aligned = cv2.warpAffine(frame, M, (w, h))

            # aligned face → 256×256
            face_256 = cv2.resize(aligned, (FACE_SIZE, FACE_SIZE))
            cv2.imwrite(str(face_dir / f"{frame_idx:05d}.png"), face_256)

            # lip landmark on 256×256
            res2 = face_mesh_static.process(
                cv2.cvtColor(face_256, cv2.COLOR_BGR2RGB)
            )
            if res2.multi_face_landmarks:
                lm2 = res2.multi_face_landmarks[0].landmark
                xs  = [int(lm2[i].x * FACE_SIZE) for i in LIP_LANDMARKS]
                ys  = [int(lm2[i].y * FACE_SIZE) for i in LIP_LANDMARKS]
                x1  = max(0,         min(xs) - PAD)
                x2  = min(FACE_SIZE, max(xs) + PAD)
                y1  = max(0,         min(ys) - PAD)
                y2  = min(FACE_SIZE, max(ys) + PAD)

                if x2 > x1 and y2 > y1:
                    lip = face_256[y1:y2, x1:x2]
                    lip = cv2.resize(lip, (LIP_SIZE, LIP_SIZE))
                    cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.png"), lip)
                    lip_meta_list.append([1, frame_idx, x1, y1, x2, y2])
                    saved_indices.append(frame_idx)
                    frame_idx += 1
                    continue

        # fallback: invalid
        cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.png"),
                    np.zeros((LIP_SIZE, LIP_SIZE, 3), np.uint8))
        cv2.imwrite(str(face_dir / f"{frame_idx:05d}.png"),
                    np.zeros((FACE_SIZE, FACE_SIZE, 3), np.uint8))
        lip_meta_list.append([0, frame_idx, 0, 0, 0, 0])
        frame_idx += 1

    cap.release()
    face_mesh_video.close()
    face_mesh_static.close()

    np.save(str(out_dir / "lip_meta.npy"),      np.array(lip_meta_list))
    np.save(str(out_dir / "frame_indices.npy"),  np.array(saved_indices))
    info(f"전처리 완료: {frame_idx}프레임 (valid={len(saved_indices)})")
    return frame_idx
