#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 02:25:54 2025


"""



import pandas as pd


merged_all = pd.read_csv("/Users/anz/Desktop/prep/fig_4/cross_sample_summary.csv")

# ==========  ==========
pair_counts_list = []
df_celltypes = merged_all.iloc[:, [2, 3]]
df_celltypes.columns = ['celltype1', 'celltype2']

df_celltypes['pair'] = df_celltypes.apply(lambda row: tuple(sorted([row['celltype1'], row['celltype2']])), axis=1)
pair_counts = df_celltypes.groupby('pair').size().reset_index(name='count')
pair_counts_list.append(pair_counts)


merged_df = pd.concat(pair_counts_list)
merged_avg = merged_df.groupby('pair')['count'].mean().reset_index()

#
merged_avg[['celltype1', 'celltype2']] = pd.DataFrame(merged_avg['pair'].tolist(), index=merged_avg.index)
merged_avg.drop(columns='pair', inplace=True)






all_celltypes = sorted(set(merged_avg['celltype1']).union(set(merged_avg['celltype2'])))
matrix = pd.DataFrame(0, index=all_celltypes, columns=all_celltypes)

# 
for _, row in merged_avg.iterrows():
    matrix.loc[row['celltype1'], row['celltype2']] = row['count']
    matrix.loc[row['celltype2'], row['celltype1']] = row['count']


bubble_df = matrix.reset_index().melt(id_vars='index')
bubble_df.columns = ['celltype1', 'celltype2', 'count']
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt


bubble_df['count_log'] = np.log1p(bubble_df['count'])  


import random
bubble_df['count_jitter'] = bubble_df['count_log'].apply(lambda x: x + random.uniform(0.1, 0.3) if x > 0 else 0)


plt.figure(figsize=(10, 9))
bubble = plt.scatter(
    x=bubble_df['celltype2'],
    y=bubble_df['celltype1'],
    s=bubble_df['count_jitter'] * 190,  
    c=bubble_df['count_log'],          
    cmap='Spectral',
    edgecolors='black'
)

plt.xticks(rotation=45)
plt.title('Average Bubble Heatmap (log-scaled): Gene Interactions Between Cell Types', fontsize=16, weight='bold')
plt.colorbar(bubble, label='log(1 + Avg. Pair Count)')
plt.grid(True)
plt.tight_layout()

#
# save_path = os.path.join(base_path, "bubble_heatmap_avg_log_2.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
plt.show()
