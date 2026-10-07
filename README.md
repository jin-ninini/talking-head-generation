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
├── data/
│   ├── HDTF_processed/      # Preprocessed HDTF videos (not included)
│   ├── splits/              # train / val / test video lists
│   └── output/
│       ├── inference/       # Generated sample videos
│       └── lse_results_ADLip2.csv
├── docs/                    # GitHub Pages demo site and poster
├── models/                  # Training runs (logs, samples, *.pt) and syncnet/ checkpoints (git-ignored)
├── src/
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
├── README.md
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
```

Or with pip (CUDA 11.8 builds):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

- Training also needs `lipsyncscore` from [lip-sync-score](https://github.com/weonjungin/lip-sync-score); put its `src/` folder on `PYTHONPATH`
- Generated videos get audio only when `ffmpeg` is on `PATH`; otherwise a silent mp4 is saved

### Data Preparation

- Put the preprocessed HDTF video folders in `data/HDTF_processed/` (git-ignored)
- Put the LatentSync checkpoint at `models/syncnet/latentsync_syncnet.pt` and the SyncLT checkpoint at `models/syncnet/synclt_best.pth`
- Run all commands from the repository root, since every path in `configs/` and the `test.py` defaults is relative to it

### Training

```bash
python train.py --cfg configs/expD3.yaml
python train.py --cfg configs/expD3.yaml --ckpt models/<run>/best.pt   # start from a checkpoint
```

- `src/trainer.py` sets `CUDA_VISIBLE_DEVICES="1"` (`src/trainer_ddp.py`: `"1,2,3"`), as on the original server; edit that line to match your GPUs
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

### Checkpoints

- `.pt` files are not in Git; download them from [Google Drive](https://drive.google.com/drive/folders/1kRF8PGcIMjcGzjbLZh6Jc6HIfCz-Z6Ek?usp=sharing) into `models/<run>/`

<br>

## Notes

- `python train.py --ddp` (run with `torchrun`) is not runnable as shipped: `src/trainer_ddp.py` imports `losses` and `utils.*`, which never existed in the original repository; it is kept for reference to the G-series runs
- `--postprocess` uses GFPGAN weight 0.5 and keeps the original chroma; a strong GFPGAN weight can "fix" real open-mouth shapes and lower sync scores
- The reference frame is chosen at random with `--seed`; the reference latent stays fixed across sliding windows to avoid jitter
- `data/output/inference/{D3,D5,F1,G1}` came from an earlier A/B/C experiment script that is no longer in the repository
- Related work: MuseTalk (Zhang et al., 2024), AttnWav2Lip (Wang et al., 2022)
