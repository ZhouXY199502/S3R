# S3R
Spatially smooth and high dimensional regression


````markdown
# 🧠 S3R: Spatially Smooth Sparse Regression

**S3R** (Spatially Smooth Sparse Regression) is a Python framework for estimating smooth and sparse regression coefficients across spatial samples — suitable for **spatial transcriptomics**, **biomedical imaging**, and other **high-dimensional spatial data**.

---

## 🔍 Overview

**S3R (Spatially Smooth Sparse Regression)** is a spatially regularized regression framework that learns **interpretable, spatially varying coefficients** from high-dimensional biological or spatial omics data.

It integrates three key principles:

1. **Data fidelity** — ensures accurate reconstruction of observed responses.  
2. **Spatial smoothness** — enforces local consistency across neighboring spatial locations.  
3. **Sparsity and group regularization** — selects a compact, biologically meaningful set of features while maintaining interpretability.

The model optimizes a composite loss function that balances these components:

<img width="547" height="80" alt="image" src="https://github.com/user-attachments/assets/2439c033-08da-4610-b3bf-e11854d35474" />


where  
- \( X \): feature matrix (samples × features)  
- \( Y \): response variable (samples × 1)  
- \( L \): spatial structure matrix (e.g., Laplacian or incidence)  
- \( W \): coefficient matrix representing spatially varying regression weights  

This formulation allows S3R to uncover **spatially coherent feature–response relationships**, identify **region-specific drivers**, and preserve **interpretability** across complex spatial domains.  
It can be applied to **spatial transcriptomics**, **histology-aligned omics**, or other **spatially indexed regression problems**.

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
| `*_##result_BT_LNORM.csv` | Estimated coefficient matrix (L-mode) |
| `*_##result_BT_HTV.csv`   | Estimated coefficient matrix (H-mode) |
| `*_##result_BT.png`       | Corresponding coefficient heatmap     |
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

> Zhou, X., Dang, P., Tang, H., Peng, L. X., Yeh, J. J., Sears, R. C., ... & Cao, S. (2025). S3R: Spatially Smooth and Sparse Regression Reveals High-Dimensional Regulatory Networks in Spatial Transcriptomics. bioRxiv, 2025-09.



---

```


```
