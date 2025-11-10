#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 00:44:56 2025

"""


import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#-----------------------------------------------------------   data loading  -----------

#############


import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

# =======    load the data parth ===============================================================================
PKL_PATH = "/Users/anz/Desktop/prep/fig_3/input_to_fig3.pkl"
# OUTPUT_PDF = "/Users/anz/Desktop/prep/fig_3/KRT14_HLADRB1_plots.pdf"




with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

df_krt = data["KRT14"].copy()
df_hla = data["HLA-DRB1"].copy()


df_krt = df_krt.rename(columns={df_krt.columns[-1]: "Keratinocytes"})

coords_df = data["KRT14"].iloc[:,:2]

def plot_panel(ax, df, gene):
    vals = df[gene].to_numpy(dtype=float)
    vmax = np.abs(vals).max() if np.any(vals != 0) else 1.0
    vmin = -vmax
    s = 20 if len(df) <= 5000 else 4

    ax.scatter(df["x"], df["y"], c=vals, cmap=custom_cmap, vmin=vmin, vmax=vmax, s=s)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    ax.invert_yaxis()
    ax.set_title(gene, fontsize=13)


# with PdfPages(OUTPUT_PDF) as pdf:
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     axes[0, 1].set_visible(False)  # 

#     plot_panel(axes[0, 0], df_krt, "Keratinocytes")
#     plot_panel(axes[1, 0], df_hla, "Macrophages")
#     plot_panel(axes[1, 1], df_hla, "DCs")

#     plt.tight_layout()
#     pdf.savefig(fig)
#     plt.close(fig)






##########     Plot   KRT14       -------------------------------------------------------------


coeff_df=  df_krt.T


import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap


feature_names_to_plot = ["Keratinocytes"]   #
# feature_names_to_plot = ["SOX10", "MYRF", "TCF4"]  

# ======== 颜色映射保持不变 ========
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    [
        (0.0, "#55C667FF"),
        (0.495, "#c7e9c0"),
        (0.505, "#c7e9c0"),
        (1.0, "#ff7f00")
    ],
    N=256
)


for title in feature_names_to_plot:



    values = coeff_df.loc[title].values.astype(float)

    # 色阶自动调节
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        vmin, vmax = -1, 1
    else:
        vmin = np.percentile(values, 7)
        vmax = np.percentile(values, 93)

    x = coords_df["x"]
    y = coords_df["y"]

    num_samples = len(x)
    scatter_size = 20 if num_samples <= 5000 else 4

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0.0)
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor("white")

    ax.scatter(x, y, c=values, cmap=custom_cmap, vmin=vmin, vmax=vmax, s=scatter_size)

    ax.set_xticks([]); ax.set_yticks([])
    ax.axis('off')
    ax.invert_yaxis()

    plt.show()
    
    

#         plot  HLA-DRB1   ------------------------------------------------------------------





coeff_df=  df_hla.T


import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap


# feature_names_to_plot = ["Keratinocytes"]   #
feature_names_to_plot = ["Macrophages", "DCs"]   # 


custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    [
        (0.0, "#55C667FF"),
        (0.495, "#c7e9c0"),
        (0.505, "#c7e9c0"),
        (1.0, "#ff7f00")
    ],
    N=256
)

# ======== 开始绘图 ========
for title in feature_names_to_plot:



    values = coeff_df.loc[title].values.astype(float)

    # 色阶自动调节
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        vmin, vmax = -1, 1
    else:
        vmin = np.percentile(values, 7)
        vmax = np.percentile(values, 93)

    x = coords_df["x"]
    y = coords_df["y"]

    num_samples = len(x)
    scatter_size = 20 if num_samples <= 5000 else 4

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0.0)
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor("white")

    ax.scatter(x, y, c=values, cmap=custom_cmap, vmin=vmin, vmax=vmax, s=scatter_size)

    ax.set_xticks([]); ax.set_yticks([])
    ax.axis('off')
    ax.invert_yaxis()

    plt.show()
