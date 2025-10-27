# S3R
Spatially smooth and high dimensional regression


````markdown
# 🧠 S3R: Spatially Smooth Sparse Regression

**S3R** (Spatially Smooth Sparse Regression) is a Python framework for estimating smooth and sparse regression coefficients across spatial samples — suitable for **spatial transcriptomics**, **biomedical imaging**, and other **high-dimensional spatial data**.

---

## 🔍 Overview

S3R integrates **sparsity**, **smoothness**, and **spatial regularization** to learn interpretable coefficient matrices that vary smoothly across spatial locations.

It supports two complementary regularization modes:

- **L-mode** — Laplacian-based smoothing using spatial adjacency.
- **H-mode** — Total Variation (TV) smoothing using graph edge structure.

Mathematically, the model minimizes:

\[
\mathcal{L}(W) = \|Y - XW\|_2^2 + \lambda \|LW\|_1 + \alpha \|W\|_1 + \beta \|W\|_{2,1}
\]

where:
- \( X \): feature matrix (samples × features)  
- \( Y \): target variable (samples × 1)  
- \( L \): normalized Laplacian or edge-incidence matrix  
- \( W \): coefficient matrix (spatially varying regression weights)

---

## ⚙️ Project Structure

| File | Description |
|------|--------------|
| `s3r_core.py` | Core implementation of S3R, including custom loss functions (`custom_loss_v4_mask`, `custom_loss_tv_mask`), graph/Laplacian construction, and Optuna-based hyperparameter search. |
| `run_s3r_exp.py` | Example main script for training and inference — performs `train_s3r()` for tuning and `run_s3r()` for final coefficient estimation. |
| `examples/` | Folder for sample datasets or test scripts. |
| `README.md` | Documentation (this file). |

---

## 🧩 Main Functions

### `train_s3r()`

Performs:
- Data loading (`X`, `Y`, `coords`)
- Graph Laplacian / edge matrix construction
- Cross-validation
- Hyperparameter optimization with **Optuna**

**Usage:**
```python
res = train_s3r(
    directory="/path/to/data/",
    X_path="/path/to/X.csv",
    Y_path="/path/to/Y.csv",
    coords_path="/path/to/coords.csv",
    NNN=1,
    loss_mode="L",        # or "H"
    n_splits=5,
    n_trials=10,
    n_jobs=4
)
````

---

### `run_s3r()`

Performs:

* Final model fitting with the best parameters
* Optionally override hyperparameters manually
* Exports coefficient matrices and visualizations

**Usage:**

```python
run_s3r(
    train_result=res,                      # or None if manual parameters
    X_path="/path/to/X.csv",
    Y_path="/path/to/Y.csv",
    coords_path="/path/to/coords.csv",
    save_dir="/path/to/results/",
    loss_mode="L",                         # or "H"
    lambda_smooth=0.01,
    alpha=0.05,
    beta=0.02
)
```

If `train_result=None`, you must manually specify `lambda_smooth`, `alpha`, and `beta`.

---

## 📊 Output Files

| File                          | Description                           |
| ----------------------------- | ------------------------------------- |
| `*_##noknockoff_BT_LNORM.csv` | Estimated coefficient matrix (L-mode) |
| `*_##noknockoff_BT_HTV.csv`   | Estimated coefficient matrix (H-mode) |
| `*_##noknockoff_BT.png`       | Corresponding coefficient heatmap     |
| `slurm-<jobid>.out`           | HPC job output log                    |

---

## 💻 HPC Execution Example

**Example SLURM script (`run_s3r_exp.sh`):**

```bash
#!/bin/bash
#SBATCH -J s3r_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=42:00:00
#SBATCH --mem=48G
#SBATCH --output=slurm-%j.out
#SBATCH -A r00077

module load python
cd /N/slate/zhou19/real_saptial/github/
python run_s3r_exp.py
```

Submit with:

```bash
sbatch run_s3r_exp.sh
```

---

## 🧰 Dependencies

* Python ≥ 3.10
* PyTorch ≥ 2.0
* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn
* Optuna

---

## 🧬 Citation

If you use S3R in your research, please cite:

> Zhou, X., Cao, S., Zhang, C.
> *Spatially Smooth Sparse Regression for High-Dimensional Spatial Transcriptomics.*
> (Manuscript in preparation)

---

## 📁 Example Output Preview

<p align="center">
  <img src="examples/example_heatmap.png" width="500"/>
</p>

---

## ✨ License

This project is distributed under the MIT License.

```

---

是否希望我帮你继续写一个小节  
👉 “**Example Usage with Real Data (Breast Cancer Spatial Dataset)**”  
用于展示一段真实路径 `/N/slate/zhou19/real_saptial/github/` 下的数据运行示例？  
这在 GitHub 页面上会让你的项目看起来更完整。
```
