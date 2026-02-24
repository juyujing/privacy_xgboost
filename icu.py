import polars as pl
import warnings
from sklearn.model_selection import GroupShuffleSplit
import os

BASE_DIR_ICU = "mimic-iv/3.1/icu"

# スキーマ
SCHEMA_ICU = {
    "subject_id": pl.Int64,
    "hadm_id": pl.Int64,
    "stay_id": pl.Int64,
    "itemid": pl.Int64,
    "valuenum": pl.Float64,
    "value": pl.String,
    "amount": pl.Float64,
    "totalamount": pl.Float64,
    "rate": pl.Float64,
    "patientweight": pl.Float64,
    "originalamount": pl.Float64,
    "originalrate": pl.Float64
}

def process_icu_with_checks():
    print("[ICU] 処理中...")

    stays = pl.scan_csv(f"{BASE_DIR_ICU}/icustays.csv.gz", schema_overrides=SCHEMA_ICU)
    charts = pl.scan_csv(f"{BASE_DIR_ICU}/chartevents.csv.gz", schema_overrides=SCHEMA_ICU)
    outputs = pl.scan_csv(f"{BASE_DIR_ICU}/outputevents.csv.gz", schema_overrides=SCHEMA_ICU)
    inputs = pl.scan_csv(f"{BASE_DIR_ICU}/inputevents.csv.gz", schema_overrides=SCHEMA_ICU)

    # 1. 脱落チェック (LOS >= 48h)
    cohort = stays.with_columns(pl.col("los").cast(pl.Float64)).filter(pl.col("los") >= 2.0)
    
    # 2. 外れ値処理
    charts_clean = charts.select(["stay_id", "charttime", "itemid", "valuenum", "valueuom"]).with_columns([
        pl.col("charttime").str.to_datetime(),
        pl.when((pl.col("itemid") == 220045) & ((pl.col("valuenum") < 20) | (pl.col("valuenum") > 300)))
          .then(None) 
          .when((pl.col("itemid") == 220210) & ((pl.col("valuenum") < 4) | (pl.col("valuenum") > 60)))
          .then(None) 
          .otherwise(pl.col("valuenum"))
          .alias("valuenum")
    ])

    final_stays = cohort.collect()
    final_charts = charts_clean.collect(engine="streaming")
    final_outputs = outputs.select(["stay_id", "charttime", "itemid", "value"]).with_columns([
        pl.col("charttime").str.to_datetime(),
        pl.col("value").cast(pl.Float64, strict=False)
    ]).collect(engine="streaming")
    
    final_inputs = inputs.select([
        "stay_id", "starttime", "endtime", "itemid", "amount", "rate", "totalamount", "patientweight"
    ]).with_columns([
        pl.col("starttime").str.to_datetime(),
        pl.col("endtime").str.to_datetime()
    ]).collect(engine="streaming")

    return final_stays, final_charts, final_outputs, final_inputs


def generate_icu_dataset(icu_stays, icu_charts, icu_outputs, icu_inputs):
    
    T_WINDOW = "24h"
    PRED_WINDOW = "24h"

    print(f"[ICU] {T_WINDOW}スライディングウィンドウのサンプリングとアノテーション...")

    weight_df = (
        icu_inputs.filter(pl.col("patientweight").is_not_null())
        .group_by("stay_id")
        .agg(pl.col("patientweight").mean().alias("weight"))
    )

    uo_hourly = (
        icu_outputs.filter(pl.col("value").is_not_null())
        .sort(["stay_id", "charttime"])
        .group_by_dynamic("charttime", every="1h", group_by="stay_id")
        .agg(pl.col("value").sum().alias("hourly_uo"))
        .join(weight_df, on="stay_id", how="inner")
        .with_columns((pl.col("hourly_uo") / pl.col("weight")).alias("uo_rate"))
    )

    # 基本アノテーション
    labels = (
        uo_hourly.sort(["stay_id", "charttime"])
        .rolling(index_column="charttime", period="6h", group_by="stay_id")
        .agg(pl.col("uo_rate").mean().alias("uo_avg_6h"))
        .with_columns(
            pl.when(pl.col("uo_avg_6h") < 0.5).then(1).otherwise(0).alias("aki_event")
        )
    )

    # 4. ラベルの生成
    core_items = {220045: "hr", 220210: "rr", 220179: "sbp", 220181: "dbp"}
    core_items_str = {str(k): v for k, v in core_items.items()}

    vitals = (
        icu_charts.filter(pl.col("itemid").is_in(list(core_items.keys())))
        .with_columns(
            pl.col("itemid").cast(pl.String).replace(core_items_str).alias("item_name")
        )
    )

    vitals_features = (
        vitals.sort(["stay_id", "charttime"])
        .group_by_dynamic("charttime", every="1h", period=T_WINDOW, group_by=["stay_id", "item_name"])
        .agg(pl.col("valuenum").mean())
        .pivot(index=["stay_id", "charttime"], on="item_name", values="valuenum")
    )

    # 特徴量の生成
    uo_features = (
        uo_hourly.sort(["stay_id", "charttime"])
        .group_by_dynamic("charttime", every="1h", period=T_WINDOW, group_by="stay_id")
        .agg([
            pl.col("uo_rate").mean().alias("feat_uo_mean"),
            pl.col("uo_rate").std().alias("feat_uo_std")
        ])
    )

    # スライディングアライメント
    features = vitals_features.join(uo_features, on=["stay_id", "charttime"], how="inner")
    
    features_sorted = features.sort(["stay_id", "charttime"]).with_columns(
        pl.col("charttime").set_sorted()
    )
    
    future_labels = (
        labels.sort(["stay_id", "charttime"])
        .rolling(index_column="charttime", period=PRED_WINDOW, group_by="stay_id")
        .agg(pl.col("aki_event").max().alias("future_aki_label"))
        .with_columns(
            pl.col("future_aki_label").shift(-1).over("stay_id")
        )
    )
    
    labels_sorted = future_labels.select(["stay_id", "charttime", "future_aki_label"]).sort(["stay_id", "charttime"]).with_columns(
        pl.col("charttime").set_sorted()
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Sortedness of columns cannot be checked.*")
        
        dataset = features_sorted.join_asof(
            labels_sorted,
            on="charttime",
            by="stay_id",
            strategy="forward"
        ).drop_nulls(subset=["future_aki_label"])

    id_map = icu_stays.select(["subject_id", "stay_id"]).unique()
    dataset = dataset.join(id_map, on="stay_id", how="inner")

    return dataset


def split_and_save_icu_datasets(dataset, output_dir="processed_data", train_ratio=0.7, tune_ratio=0.15, test_ratio=0.15):
    """
    Train, Tune, Test に分割する。
    """
    print(f"ICU データセットの分割 (Train:{train_ratio}, Tune:{tune_ratio}, Test:{test_ratio})...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    groups = dataset["subject_id"].to_numpy()
    
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_tune_idx, test_idx = next(gss_test.split(dataset, groups=groups))
    
    test_data = dataset.filter(pl.arange(0, pl.len()).is_in(test_idx))
    train_tune_df = dataset.filter(pl.arange(0, pl.len()).is_in(train_tune_idx))

    tune_relative_size = tune_ratio / (train_ratio + tune_ratio)
    gss_tune = GroupShuffleSplit(n_splits=1, test_size=tune_relative_size, random_state=42)
    
    train_idx, tune_idx = next(gss_tune.split(train_tune_df, groups=train_tune_df["subject_id"]))
    
    train_data = train_tune_df.filter(pl.arange(0, pl.len()).is_in(train_idx))
    tune_data = train_tune_df.filter(pl.arange(0, pl.len()).is_in(tune_idx))

    # 保存
    files = {"train": train_data, "tune": tune_data, "test": test_data}
    
    for name, df in files.items():
        path = os.path.join(output_dir, f"icu_{name}.parquet")
        df.write_parquet(path)
        print(f"ICU {name.upper()} セットの保存先: {path} (サンプル数: {len(df)}, 患者数: {df['subject_id'].n_unique()})")

    return train_data, tune_data, test_data


if __name__ == "__main__":
    i_stays, i_charts, i_outputs, i_inputs = process_icu_with_checks()
    i_dataset = generate_icu_dataset(i_stays, i_charts, i_outputs, i_inputs)
    
    print(f"\nICUデータセットのサンプル数: {len(i_dataset)}")
    print(f"AKI ラベルの分布: \n{i_dataset['future_aki_label'].value_counts()}")
    
    train_df, tune_df, test_df = split_and_save_icu_datasets(i_dataset)
    print("Done.")