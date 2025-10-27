import sys
sys.path.append("/N/slate/zhou19/real_saptial/github/")
from s3r_core import train_s3r, run_s3r


# import sys
# sys.path.append("/N/slate/zhou19/real_saptial/github/")
# from s3r_core import train_s3r, run_s3r

# def train_s3r(
#     directory: str,
#     NNN: int = 1,
#     loss_mode: str = "L",
#     n_splits: int = 5,
#     random_seed: int = 42,
    # X_path="/Users/a/Downloads/breast_cancer_new_2/1_new_X.csv",
    # Y_path="/Users/a/Downloads/breast_cancer_new_2/1_new_Y.csv",
    # coords_path="/Users/a/Downloads/breast_cancer_new_2/1_coords.csv",
#     n_trials: int = 50,
#     n_jobs: int = 4,
#     learning_rate: float = 1e-3,
#     num_iterations: int = 30000,
#     convergence_threshold: float = 1e-4,
#     convergence_threshold2: float = 2e-3,
#     max_convergence_count: int = 20,
#     knn_k: int = 7,
#     topk_features: int = 50,
#     verbose: bool = True,
# ) -> dict



# def run_s3r(
#     train_result: dict,
#     save_dir: str,
#     loss_mode: str = "L",
#     device_name: str = "cuda",
#     post_lr: float = 3e-4,
#     post_iters: int = 100000,
#     early_start_iter: int = 5000,
#     early_window: int = 1000,
#     early_eps: float = 1e-14,
#     fig_size=(15, 12),
#     lambda_smooth=0.002391076717339003,
#     alpha=0.035065877784368496,
#     beta=0.03960085437733601
# ) -> dict

if __name__ == "__main__":
    # 1) hyperparameter tuning
    res = train_s3r(
        directory="/N/slate/zhou19/real_saptial/github/",
        X_path="/N/slate/zhou19/real_saptial/github/X_exp.csv",
        Y_path="/N/slate/zhou19/real_saptial/github/Y_exp.csv",
        coords_path="/N/slate/zhou19/real_saptial/github/C_exp.csv",
        NNN=1,
        loss_mode="L",       # or "H"
        n_splits=5,
        random_seed=42,
        n_trials=30,
        n_jobs=4
    )

    # 2) post-optimization run and export

    run_s3r(
    train_result=res,
        X_path="/N/slate/zhou19/real_saptial/github/X_exp.csv",
        Y_path="/N/slate/zhou19/real_saptial/github/Y_exp.csv",
        coords_path="/N/slate/zhou19/real_saptial/github/C_exp.csv",
    save_dir="/N/slate/zhou19/real_saptial/github/results000",
    loss_mode="L",

)


