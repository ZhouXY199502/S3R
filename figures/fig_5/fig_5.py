#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 03:34:22 2025


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 输入与目标 =========
csv_path = "/Users/anz/Desktop/prep/fig_5/KRT14.csv"      #   load your data    ERBB2.csv   or KRT14.csv
out_dir  = "/Users/anz/Desktop/prep/fig_5"
os.makedirs(out_dir, exist_ok=True)

gene    = "KRT14"     #  change to target gene
feature = "KRT16"     #    #   change to feature


def _autosize(n):
    if n <= 2000:  return 6
    if n <= 5000:  return 3
    if n <= 30000: return 1.2
    return 0.6

def plot_spatial(values, x, y, out_png, cmap="Reds", vmin=None, vmax=None, transparent=True):
    plt.figure(figsize=(4, 3), dpi=300)
    s = _autosize(len(values))
    plt.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                s=s, edgecolors='none', alpha=0.95)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0,
                transparent=transparent)
    plt.close()
    print(f"✅ saved: {out_png}")


df = pd.read_csv(csv_path, index_col=0)


coords_df = pd.DataFrame({"x": df.loc["x"], "y": df.loc["y"]}).astype(float)


coeff_df = df.drop(index=["x", "y"])


values = coeff_df.loc[feature, coords_df.index].astype(float).values
values_pos = values.copy()
values_pos[values_pos < 0] = np.nan


if np.any(np.isfinite(values_pos)):
    vmin = 0.0
    vmax = np.nanpercentile(values_pos, 95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(values_pos) if np.isfinite(np.nanmax(values_pos)) else 1.0
else:
    vmin, vmax = 0.0, 1.0

out_png_feat = os.path.join(out_dir, f"{gene}_{feature}_spatial.png")
plot_spatial(values_pos, coords_df["x"].values, coords_df["y"].values,
             out_png=out_png_feat, cmap="Reds", vmin=vmin, vmax=vmax, transparent=True)
