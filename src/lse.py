"""LSE-C / LSE-D of generated videos with syncnet_python (joonson/syncnet_python on GitHub).

Runs its run_pipeline.py (face tracks) and run_syncnet.py on each video, then averages the per-track
confidence (LSE-C) and min distance (LSE-D) weighted by track length.
"""
import os
import pickle
import subprocess
import sys

import numpy as np


def lse(video, name, syncnet_dir, work_dir):
    for script in ("run_pipeline.py", "run_syncnet.py"):
        subprocess.run([sys.executable, script, "--videofile", os.path.abspath(video), "--reference", name,
                        "--data_dir", os.path.abspath(work_dir)], cwd=syncnet_dir, check=True)
    with open(os.path.join(work_dir, "pywork", name, "activesd.pckl"), "rb") as f:
        tracks = pickle.load(f)  # per track: [frames, 2*vshift+1] distances

    confs, dists, lens = [], [], []
    for d in tracks:
        mdist = np.asarray(d).mean(0)
        confs.append(np.median(mdist) - mdist.min())
        dists.append(mdist.min())
        lens.append(len(d))
    return float(np.average(confs, weights=lens)), float(np.average(dists, weights=lens)), len(tracks)
