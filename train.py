import os
os.environ["XGBOOST_CUDA_ALLOCATOR"] = "cuda_malloc"
import polars as pl
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import json
import warnings
import gc

warnings.filterwarnings("ignore")

def load_data(prefix="hosp", split="train", data_dir="processed_data"):
    # Parquet データの読み込み
    path = os.path.join(data_dir, f"{prefix}_{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Data not found.")
        
    print(f"{split.upper()} データセットの読み込み: {path}")
    df = pl.read_parquet(path)
    
    possible_id_cols = ["subject_id", "hadm_id", "stay_id", "charttime", "future_aki_label"]
    drop_cols = [c for c in possible_id_cols if c in df.columns]
    
    X = df.drop(drop_cols).to_numpy()
    y = df["future_aki_label"].to_numpy()
    
    return X, y

def objective(trial, dtrain, dtune, scale_pos_weight):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr", 
        "scale_pos_weight": scale_pos_weight,
        
        "tree_method": "hist",  
        "device": "cuda",       
        
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5)
    }

    # Optuna プルーニング
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "tune-aucpr")
    
    bst = xgb.train(
        param, 
        dtrain, 
        num_boost_round=500,
        evals=[(dtrain, "train"), (dtune, "tune")],
        early_stopping_rounds=30,
        callbacks=[pruning_callback],
        verbose_eval=False
    )
    
    best_score = bst.best_score
    
    del bst
    del pruning_callback
    gc.collect()
    
    return best_score

def evaluate_model(model, dtest, y_test):
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob >= 0.54).astype(int)

    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)

    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC:  {auprc:.4f}")
    print("-" * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No AKI (0)", "AKI (1)"]))
    print("-" * 50)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"混同行列:\n真の No AKI (TN): {cm[0][0]} | 誤判定の AKI (FP): {cm[0][1]}")
    print(f"見逃しの AKI (FN): {cm[1][0]} | 予測成功  (TP): {cm[1][1]}")
    print("="*50)

if __name__ == "__main__":

    PREFIX = "hosp"  
    
    print(f"{PREFIX.upper()}")

    X_train, y_train = load_data(PREFIX, "train")
    X_tune, y_tune = load_data(PREFIX, "tune")
    X_test, y_test = load_data(PREFIX, "test")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtune = xgb.DMatrix(X_tune, label=y_tune)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # サンプル不均衡の処理
    scale_pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    print(f"\nサンプルの分布: 負例/正例の割合 = {scale_pos_weight:.2f}")

    # Optuna ハイパーパラメータのチューニング
    print("\nOptuna ハイパーパラメータのチューニング...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective(trial, dtrain, dtune, scale_pos_weight), 
        n_trials=100, 
        gc_after_trial=True
    )

    print("\nチューニング完了。")
    best_params = study.best_params
    
    params_path = f"best_params_{PREFIX}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"ハイパーパラメータの保存先: {params_path}")

    print("\n学習...")
    
    ckpt_dir = f"checkpoints_{PREFIX}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_callback = xgb.callback.TrainingCheckPoint(
        directory=ckpt_dir,
        interval=50, 
        name="model_iter" 
    )

    final_params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": "cuda",
        
        **best_params
    }
    
    final_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dtune, "tune")],
        early_stopping_rounds=50,
        callbacks=[ckpt_callback],
        verbose_eval=50
    )

    # 評価
    evaluate_model(final_model, dtest, y_test)
    
    # 最終モデルの保存
    best_iteration = final_model.best_iteration
    model_path = f"xgboost_{PREFIX}_best_model_iter{best_iteration}.json"
    final_model.save_model(model_path)
    
    print(f"\nDone.")
    print(f"(第 {best_iteration} ラウンド) パラメータの保存先: {model_path}")