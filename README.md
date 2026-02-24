# MIMIC-IV AKI Prediction Pipeline

## File Structure

* `hosp.py` : Processes MIMIC-IV hospital data and generates the `hosp` dataset.
* `icu.py` : Processes MIMIC-IV ICU data and generates the `icu` dataset.
* `train.py` : Tunes hyperparameters (Optuna) and trains the model (XGBoost).
* `processed_data/` : Output directory for the generated `.parquet` datasets (Train/Tune/Test).
* `checkpoints_*/` : Output directory for XGBoost training checkpoints.
* `best_params_*.json` : Saved best hyperparameters from Optuna.
* `xgboost_*_best_model_iter*.json` : The final trained XGBoost model.

## Execution Flow

**1. Data Preparation**
Ensure MIMIC-IV v3.1 raw `.csv.gz` files are placed in the following directories:
* `mimic-iv/3.1/hosp/`
* `mimic-iv/3.1/icu/`

**2. Generate Datasets**
Run the data processing scripts to create the train, tune, and test splits:
```bash
python hosp.py
# and/or
python icu.py
```

**3. Train the Model
Open train.py, set the target prefix (PREFIX = "hosp" or PREFIX = "icu"), and execute:
```bash
python train.py
```
