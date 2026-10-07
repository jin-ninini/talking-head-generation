# 📌 ADLip2

> Audio-driven lip-sync generation that edits only the 96×96 lip region in VAE latent space, trained with a lip-level sync loss (SyncLT).

**[Project Demo Page](https://weonjungin.github.io/ADLip2/)** | **[Full Poster](docs/poster.jpg)**

<br>

## Overview

- Lip-sync models such as MuseTalk learn synchronization over the whole lower face, so the lip alignment signal gets diluted and LSE-C stays low
- ADLip2 tests the hypothesis that training on the lip region alone gives more precise synchronization
- The project combines a lip-only generator (**ADLip**) with **SyncLT**, a loss that learns synchronization directly in lip latent space
- Team capstone project of four members (original repository: [weonjungin/ADLip2](https://github.com/weonjungin/ADLip2))

<br>

## Pipeline

A new input is a **face video** (whose lips are re-generated) plus a **speech audio file** (what the face should say). Every step below was run on the files in `data/examples/`; the exact commands are in [Demo: Full Pipeline](#demo-full-pipeline).

```text
face video ──► preprocess.py video ──► frames/ face_frames/ lip_meta.npy frame_indices.npy
     │
     └───────► preprocess.py audio ──► audio.wav audio_feature.npy mel_latentsync.npy
speech audio ► preprocess.py audio ──► audio.wav audio_feature.npy mel_latentsync.npy

train.py (face video folders: all 7 items)  ──► models/<run>/best.pt
valid.py (val folders + best.pt)            ──► latent L1
test.py  (face video folder + speech folder + best.pt) ──► lip-synced mp4
```

**1. Video preprocessing** (`preprocess.py video`, `src/preprocess_video.py`)
- MediaPipe Face Mesh finds the face in every frame; the frame is rotated so the eyes are level (affine alignment) and resized to 256×256 → `face_frames/00000.png …`
- On that 256×256 face, the 20 lip landmarks plus a 10 px margin give the lip box, resized to 96×96 → `frames/00000.png …`
- `lip_meta.npy` `[N, 6]` = `[valid, frame_idx, x1, y1, x2, y2]` (box in 256×256 face coordinates); frames with no face are saved black with `valid = 0`
- `frame_indices.npy` = indices of the valid frames

**2. Audio preprocessing** (`preprocess.py audio`, `src/preprocess_audio.py`); the source can be a video or an audio file
- `audio.wav`: 16 kHz mono (ffmpeg)
- `audio_feature.npy` `[N, 384]`: Whisper-tiny encoder states (50 Hz), one per 25 fps video frame (token `2i` for frame `i`)
- `mel_latentsync.npy` `[80, T]`: LatentSync mel-spectrogram, 80 frames/s; the trainer cuts 52-column windows from it for the LatentSync SyncNet loss, the SyncLT loss, and the mouth discriminator input

**3. Training** (`train.py`)
- `LipSyncDataset` samples 16-frame windows where every frame is valid: target lips, two reference windows (≥ 50 frames away), audio windows `[16, 5, 384]`, the matching mel window, and the lower half of the face frames
- The generator edits reference lip latents toward the audio; the losses and stages are described in [Approach](#approach)

**4. Validation** (`valid.py`): latent L1 between the prediction (reference A + true audio) and the ground-truth lips on the val split

**5. Testing** (`test.py`)
- One reference frame of the face video is picked with `--seed`; its lip latent is the reference for every window, and its face image is the background for the whole output, so **the output face is a still image whose lips move**
- The speech features are processed in 16-frame (0.64 s) windows; any tail shorter than 16 frames is dropped
- Each predicted 96×96 lip is pasted back into the 256×256 face at the `lip_meta` box (feathered blend, or Poisson + GFPGAN with `--postprocess`)
- Output: `simple_ref<N>.mp4`, 512×256 (left: generated lips upscaled; right: face with pasted lips), muxed with `--audio_dir1/audio.wav`

<br>

## Approach

- **Data**: HDTF videos preprocessed into per-video folders (lip frames, face frames, `lip_meta.npy`, `audio_feature.npy`, mel, `audio.wav`); split lists in `data/splits/` (train 318, val 39, test 45)
- **Model**: A frozen SD VAE (`stabilityai/sd-vae-ft-mse`) encodes 16 reference lip frames; the generator (2 spatial cross-attention + 2 temporal self-attention layers) predicts only a delta, `z_out = z_ref + α·delta` (α = 0.35), which the frozen VAE decodes
- **Losses**: Latent reconstruction terms, LatentSync loss (face-level, frozen), SyncLT loss (lip-level, from [lip-sync-score](https://github.com/weonjungin/lip-sync-score)), and an optional mouth discriminator (GAN)
- **Training Stages**: Stage 1 reconstruction → Stage 2 adds LatentSync → Stage 3 adds GAN and SyncLT, each stage switched on by the loss weights in `configs/`
- **Evaluation**: Latent L1 on the val split (`valid.py`); LSE-C / LSE-D on generated videos with the external [syncnet_python](https://github.com/joonson/syncnet_python)

```text
Ref Lip (16 × 96×96)                 Audio features
        │                                  │
  VAE Encoder (frozen)                     │
        └────────────► Generator ◄─────────┘
                Spatial Cross-Attention ×2
                Temporal Self-Attention ×2
                Delta Head
                       │
               VAE Decoder (frozen)
                       │
            Generated Lip Frames (16)
```

<br>

## Results

Loss ablation, as reported by the team on the test videos:

| Condition | LatentSync | SyncLT | LSE-C ↑ | LSE-D ↓ |
|---|---|---|---|---|
| L1 (no SyncNet) | ✗ | ✗ | 0.083 | 14.983 |
| LatentSync only | ✓ | ✗ | 5.472 | 9.071 |
| SyncLT only | ✗ | ✓ | 1.898 | 12.146 |
| **LatentSync + SyncLT** | ✓ | ✓ | **5.966** | **8.583** |

- LatentSync (face level) and SyncLT (lip level) give complementary signals; combining them gives the best LSE-C and LSE-D
- Lip information alone (SyncLT only) is not enough to learn synchronization
- `expD3` (`lambda_our=0.15`, `our_lambda_neg=2.5`) gives the most balanced SyncLT / LatentSync gradient ratio
- `expF1` (GAN + SyncLT) scores lower on LSE but looks most natural, so LSE alone does not capture visual quality
- LSE-C / LSE-D for every evaluated run are in `data/output/lse_results_ADLip2.csv`

<br>

## Tech Stack

| Category | Stack |
|---|---|
| Languages | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square\&logo=python\&logoColor=white) |
| Data Analysis | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white) |
| Deep Learning | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white) |
| NLP & LLM | ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square\&logo=huggingface\&logoColor=black) |
| Development & Environment | ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square\&logo=git\&logoColor=white)  ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square\&logo=github\&logoColor=white)  ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square\&logo=linux\&logoColor=black)  ![Conda](https://img.shields.io/badge/Conda-44A833?style=flat-square\&logo=anaconda\&logoColor=white) |

<br>

## Project Structure

```text
talking-head-generation/
├── configs/                 # Experiment configs (expD3, expD5, expF1: single GPU; expG1–G3: DDP)
│   ├── demo.yaml            # Full-pipeline demo on data/examples (expG3 settings, 1 epoch)
│   └── audio.yaml           # LatentSync mel settings
├── data/
│   ├── HDTF_processed/      # Preprocessed HDTF videos (not included)
│   ├── splits/              # train / val / test video lists
│   ├── examples/            # Demo inputs: 00001.mp4 (face video), eng_audio.wav (speech)
│   ├── demo_splits/         # Split lists for configs/demo.yaml
│   ├── demo/                # Demo preprocessing and outputs (git-ignored)
│   └── output/
│       ├── inference/       # Generated sample videos
│       └── lse_results_ADLip2.csv
├── docs/                    # GitHub Pages demo site and poster
├── models/                  # Training runs (logs, samples, *.pt) and syncnet/ checkpoints (git-ignored)
├── src/
│   ├── preprocess_video.py  # Face / lip crops (from ADLip)
│   ├── preprocess_audio.py  # 16 kHz wav, Whisper features, LatentSync mel
│   ├── latentsync_audio.py  # LatentSync mel-spectrogram (from LatentSync)
│   ├── dataset.py           # LipSyncDataset and audio windowing
│   ├── generator.py         # TemporalLipGenerator
│   ├── discriminator.py     # Mouth discriminator (GAN)
│   ├── latentsync.py        # LatentSync SyncNet (frozen, face level)
│   ├── vae_wrapper.py       # Frozen SD VAE encode / decode
│   ├── model_loading.py     # VAE, generator, and LatentSync loading
│   ├── losses.py            # Mouth crop, Sobel edge, contrastive sync loss, lip paste
│   ├── visualization.py     # Training sample images
│   ├── validation.py        # Latent L1 validation
│   ├── seed.py              # Reproducibility
│   ├── trainer.py           # Single-GPU training (AMP)
│   ├── trainer_ddp.py       # Multi-GPU DDP training (not runnable, see Notes)
│   └── inference.py         # Video generation and post-processing
├── environment.yml
├── requirements.txt
├── requirements-preprocess.txt   # mediapipe env for `preprocess.py video`
├── README.md
├── preprocess.py            # Preprocessing entry point (video / audio)
├── train.py                 # Training entry point
├── valid.py                 # Validation entry point
└── test.py                  # Testing (video generation) entry point
```

<br>

## Getting Started

### Installation

```bash
conda env create -f environment.yml
conda activate adlip
pip install transformers==5.19.0 librosa==0.11.0   # needed by preprocess.py audio (not in environment.yml)
```

Or with pip (CUDA 11.8 builds):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

`preprocess.py video` needs MediaPipe with the legacy `mp.solutions` API, which requires numpy < 2, so it runs in a separate env:

```bash
conda create -n adlip-prep -c conda-forge python=3.10 ffmpeg
conda run -n adlip-prep pip install -r requirements-preprocess.txt
```

- Everything else (`preprocess.py audio`, `train.py`, `valid.py`, `test.py`) runs in the main env
- `ffmpeg` must be on `PATH` for `preprocess.py audio` and for audio in `test.py` output (a system ffmpeg, or append the adlip-prep env's `bin/` to the end of `PATH`; prepending it makes `python` resolve to that env)

- Training also needs `lipsyncscore` from [lip-sync-score](https://github.com/weonjungin/lip-sync-score); put its `src/` folder on `PYTHONPATH`
- Generated videos get audio only when `ffmpeg` is on `PATH`; otherwise a silent mp4 is saved
- `test.py --postprocess` imports `gfpgan`, which fails with torchvision ≥ 0.17 (`No module named 'torchvision.transforms.functional_tensor'`). Fix it in the installed `basicsr/data/degradations.py` by changing that import to `torchvision.transforms.functional`

### Data Preparation

- Put the preprocessed HDTF video folders in `data/HDTF_processed/` (git-ignored), or build them from raw clips with `preprocess.py video` + `preprocess.py audio` (same output folder per clip)
- Put the LatentSync checkpoint at `models/syncnet/latentsync_syncnet.pt` and the SyncLT checkpoint at `models/syncnet/synclt_best.pth`
- Run all commands from the repository root, since every path in `configs/` and the `test.py` defaults is relative to it

### Training

```bash
python train.py --cfg configs/expD3.yaml
python train.py --cfg configs/expD3.yaml --ckpt models/<run>/best.pt   # start from a checkpoint
```

- `src/trainer.py` uses GPU `1` unless `CUDA_VISIBLE_DEVICES` is already set (e.g. `CUDA_VISIBLE_DEVICES=0 python train.py ...`); `src/trainer_ddp.py` still hard-codes `"1,2,3"`
- Training samples and in-training validation run every `sample.save_every` epochs; the sample images use three fixed HDTF clips and are skipped when `data.root` does not contain them
- Each run writes `train.log`, `samples/`, `tb/`, `last.pt`, and `best.pt` to `models/<run_name>_<timestamp>/`

### Validation

```bash
python valid.py --cfg configs/expD3.yaml --ckpt models/<run>/best.pt
```

- Prints the latent L1 on the val split, the same metric logged during training

### Testing

```bash
python test.py --ref_dir <reference video folder> --audio_dir1 <audio video folder> \
    --cfg configs/expG3.yaml --ckpt models/<run>/best.pt --out data/output/inference/G3_test
python test.py ... --postprocess   # Poisson blending + GFPGAN
```

- Writes `simple_ref<N>.mp4` (raw) or `out_simple.mp4` (`--postprocess`) and `eval.log` to `--out`
- `--test` renders only the first 16-frame window; `--max_windows` limits the number of windows
- LSE-C / LSE-D are computed separately with `run_pipeline.py` and `run_syncnet.py` from [syncnet_python](https://github.com/joonson/syncnet_python)

### Demo: Full Pipeline

Runs every step on `data/examples/` with the released G3 checkpoint (`models/G3_20260529_104341/best.pt`, see [Checkpoints](#checkpoints)). `configs/demo.yaml` is `expG3.yaml` with `data.root: data/demo`, one video in every split (`data/demo_splits/`), 10 samples per video, 1 epoch, `num_workers: 0`, `results.root: models/demo`, and `save_every: 1`, so training only proves the loop runs.

```bash
# 1. Preprocess the face video (adlip-prep env) and both audio sources (main env)
conda run -n adlip-prep python preprocess.py video --video data/examples/00001.mp4 --out data/demo/00001
python preprocess.py audio --src data/examples/00001.mp4     --out data/demo/00001
python preprocess.py audio --src data/examples/eng_audio.wav --out data/demo/eng_audio

# 2. Train (1 epoch) → models/demo/G3_demo_<timestamp>/
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/demo.yaml

# 3. Validate the released checkpoint on the demo split
python valid.py --cfg configs/demo.yaml --ckpt models/G3_20260529_104341/best.pt

# 4. Generate: face of 00001.mp4 speaking eng_audio.wav
python test.py --ref_dir data/demo/00001 --audio_dir1 data/demo/eng_audio --audio_dir2 data/demo/eng_audio \
    --cfg configs/demo.yaml --ckpt models/G3_20260529_104341/best.pt --out data/demo/output
```

What the demo produced on one RTX A6000:

| Step | Result |
|---|---|
| `preprocess.py video` | 214 frames, 213 valid (frame 170 has no face) |
| `preprocess.py audio` | `00001`: features `[214, 384]`, mel `[80, 685]`; `eng_audio` (60 s): features `[1500, 384]`, mel `[80, 4801]` |
| `train.py` | 1 epoch, train loss 1.78, in-training val 2.50; writes `best.pt` / `last.pt` |
| `valid.py` (G3 `best.pt`) | latent L1 2.03 (on the training clip itself; not a benchmark) |
| `test.py` | `simple_ref6.mp4`: 1488 frames (93 × 16), 59.52 s, video + audio |

<br>

### Checkpoints

- `.pt` files are not in Git; download them from [Google Drive](https://drive.google.com/drive/folders/1kRF8PGcIMjcGzjbLZh6Jc6HIfCz-Z6Ek?usp=sharing) into `models/<run>/`

<br>

## Notes

- `python train.py --ddp` (run with `torchrun`) is not runnable as shipped: `src/trainer_ddp.py` imports `losses` and `utils.*`, which never existed in the original repository; it is kept for reference to the G-series runs
- `--postprocess` uses GFPGAN weight 0.5 and keeps the original chroma; a strong GFPGAN weight can "fix" real open-mouth shapes and lower sync scores
- The reference frame is chosen at random with `--seed`; the reference latent stays fixed across sliding windows to avoid jitter
- `data/output/inference/{D3,D5,F1,G1}` came from an earlier A/B/C experiment script that is no longer in the repository
- `mel_latentsync.npy` was produced by a script that is not in any of the team repositories. `src/preprocess_audio.py` rebuilds it with LatentSync's `melspectrogram` (`src/latentsync_audio.py`, `configs/audio.yaml`), which matches how the file is read (80 mels, 80 frames/s, 52-frame windows fed to LatentSync `get_audio_embed`), but it is not confirmed to be byte-identical to the original HDTF files
- `preprocess.py video` reproduces the ADLip preprocessing byte for byte; `audio_feature.npy` from `preprocess.py audio` differs from the ADLip sample by at most 0.04 (mean 1.7e-5), from the GPU and `transformers` version
- The SyncLT checkpoint contains temporal GRU weights, but `src/trainer.py` builds `SyncNetTemporal` without them and loads with `strict=False`, so the GRU is dropped (same as the original training code)
- When a video has invalid frames, `frames/` is indexed by frame position while `frame_indices.npy` lists only valid frames, so the mel window for frames after the first invalid one is shifted by the number of invalid frames before it, and the last frames of such a clip are never sampled (original behavior)
- Code sources: `src/preprocess_video.py` and `src/preprocess_audio.py` from [ADLip](https://github.com/Inha-Jeong/ADLip), `src/latentsync_audio.py` and `configs/audio.yaml` from [LatentSync](https://github.com/bytedance/LatentSync) (Apache-2.0), SyncLT from [lip-sync-score](https://github.com/weonjungin/lip-sync-score)
- Related work: MuseTalk (Zhang et al., 2024), AttnWav2Lip (Wang et al., 2022)
