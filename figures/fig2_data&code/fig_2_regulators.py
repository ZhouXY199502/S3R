#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 23:19:52 2025

@author: xinyu zhou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#-----------------------------------------------------------   data loading  -----------
Sample_id = "151673"      #   151673    151507
chosen_base = "MBP"       # target genes
feature = "MYRF"           # regulator features

BASE_DIR = "/Users/anz/Desktop/prep/fig_2"             # You must replace it with your file storage path.
file_path = os.path.join(BASE_DIR, "input_to_fig2.pkl")


with open(file_path, "rb") as f:
    data = pickle.load(f)


boundary_path = os.path.join(BASE_DIR, f"outside_{Sample_id}_no_H&E.csv")   ######  This is a boundary annotation between different layers.
boundary_df = pd.read_csv(boundary_path)
boundary_df["y_new"] = -boundary_df["y_new"]  


#####--------------------------------------------------------
coords_df = data[f"{Sample_id}_coords"].copy()
df = data[Sample_id][chosen_base].copy()   
coeff_values = df[feature].to_numpy(dtype=float)   


x_coord = coords_df['col']
y_coord = coords_df['row']


vals = coeff_values
# vmax = np.abs(vals).max() if np.any(vals != 0) else 1.0
# vmin = -vmax
low, high = np.percentile(vals[np.isfinite(vals)], [1, 99])  # 1%–99% 
# 防止 low == high
if high - low < 1e-9:
    vmin, vmax = -1, 1
else:
    # 为保持中点对称性，取绝对最大
    abs_max = max(abs(low), abs(high))
    vmin, vmax = -abs_max, abs_max
    
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_blue_white_red',
    ['#b3b3b3', '#b3b3b3', 'chartreuse']
)

plt.figure(figsize=(6, 6))
plt.scatter(x_coord, y_coord, c=vals, cmap=custom_cmap,
            vmin=vmin, vmax=vmax, s=8)


plt.scatter(boundary_df['x_new'], boundary_df['y_new'],
            color='red', s=8, edgecolors='black', linewidths=0.2)


plt.grid(False); plt.xticks([]); plt.yticks([])
ax = plt.gca()
for side in ['top','right','left','bottom']:
    ax.spines[side].set_visible(False)
ax.invert_yaxis()
ax.set_facecolor('none')
plt.title(f"{Sample_id}_{chosen_base}_{feature}", fontsize=13)

output_path = f"{BASE_DIR}/{Sample_id}_{chosen_base}_{feature}_regulators.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
plt.show()

print("Saved:", output_path)
