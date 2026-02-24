import polars as pl
import warnings
from sklearn.model_selection import GroupShuffleSplit
import os

BASE_DIR_HOSP = "mimic-iv/3.1/hosp"
ID_SCHEMA = {"subject_id": pl.Int64, "hadm_id": pl.Int64, "itemid": pl.Int64}

def process_hosp_with_checks():
    print("[HOSP] 処理中...")
    
    # データの読み込み
    pts = pl.scan_csv(f"{BASE_DIR_HOSP}/patients.csv.gz", schema_overrides=ID_SCHEMA)
    adm = pl.scan_csv(f"{BASE_DIR_HOSP}/admissions.csv.gz", schema_overrides=ID_SCHEMA)
    diag = pl.scan_csv(f"{BASE_DIR_HOSP}/diagnoses_icd.csv.gz", schema_overrides=ID_SCHEMA)
    labs = pl.scan_csv(f"{BASE_DIR_HOSP}/labevents.csv.gz", schema_overrides={**ID_SCHEMA, "valuenum": pl.Float64})
    
    # 基準クレアチニンの計算 (Baseline SCr)
    scr_labs = labs.filter(pl.col("itemid") == 50912).select(
        ["subject_id", "charttime", "valuenum"]
    ).with_columns(pl.col("charttime").str.to_datetime())
    
    admissions_dt = adm.select(["subject_id", "hadm_id", "admittime"]).with_columns(
        pl.col("admittime").str.to_datetime()
    )
    
    scr_joined = admissions_dt.join(scr_labs, on="subject_id", how="left")
    
    baseline_prior = (
        scr_joined.filter(
            (pl.col("charttime") <= pl.col("admittime")) & 
            (pl.col("charttime") >= (pl.col("admittime") - pl.duration(days=7)))
        )
        .group_by("hadm_id")
        .agg(pl.col("valuenum").min().alias("baseline_prior"))
    )
    
    baseline_first = (
        scr_joined.filter(pl.col("charttime") >= pl.col("admittime"))
        .sort(["hadm_id", "charttime"])
        .group_by("hadm_id")
        .agg(pl.col("valuenum").first().alias("baseline_first"))
    )
    
    baseline_df = (
        admissions_dt.select(["subject_id", "hadm_id"])
        .join(baseline_prior, on="hadm_id", how="left")
        .join(baseline_first, on="hadm_id", how="left")
        .with_columns(
            pl.coalesce(["baseline_prior", "baseline_first"]).alias("baseline_scr")
        )
        .filter(
            pl.col("baseline_scr").is_not_null() & 
            (pl.col("baseline_scr") >= 0.2) & 
            (pl.col("baseline_scr") <= 30.0)
        )
        .select(["subject_id", "hadm_id", "baseline_scr"])
    )

    # ESRD 患者を除外
    esrd_pts = diag.filter(pl.col("icd_code").is_in(["5855", "5856", "N185", "N186"])).select("subject_id").unique()
    
    cohort = (
        adm.join(pts.select(["subject_id", "gender", "anchor_age", "anchor_year"]), on="subject_id")
        .with_columns([
            (pl.col("anchor_age").cast(pl.Int32) + 
             (pl.col("admittime").str.to_datetime().dt.year() - pl.col("anchor_year").cast(pl.Int32))
            ).alias("age")
        ])
        .filter(pl.col("age") >= 18)
        .join(esrd_pts, on="subject_id", how="anti")
        .join(baseline_df, on=["subject_id", "hadm_id"], how="inner") 
    )

    labs_clean = labs.select(["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]).with_columns([
        pl.col("charttime").str.to_datetime(),
        pl.when((pl.col("itemid") == 50912) & ((pl.col("valuenum") <= 0) | (pl.col("valuenum") > 30)))
        .then(None).otherwise(pl.col("valuenum")).alias("valuenum")
    ])

    return cohort.collect(), labs_clean.collect(engine="streaming")


def generate_hosp_dataset(hosp_pts, hosp_labs):
    
    T_WINDOW = "24h"
    PRED_WINDOW = "24h"


    print(f"[HOSP] {T_WINDOW}スライディングウィンドウのサンプリングとアノテーション...")

    hosp_labs_inpatient = hosp_labs.filter(pl.col("hadm_id").is_not_null())

    # 基本アノテーション (AKI Event)
    scr = (
        hosp_labs_inpatient.filter(pl.col("itemid") == 50912)
        .join(hosp_pts.select(["hadm_id", "baseline_scr"]), on="hadm_id")
        .sort(["hadm_id", "charttime"])
    )

    labels = (
        scr.rolling(index_column="charttime", period="48h", group_by="hadm_id")
        .agg([
            pl.col("valuenum").min().alias("min_scr_48h"),
            pl.col("valuenum").last().alias("curr_scr"),
            pl.col("baseline_scr").first().alias("base_scr")
        ])
        .with_columns(
            pl.when((pl.col("curr_scr") >= pl.col("min_scr_48h") + 0.3) | 
                    (pl.col("curr_scr") >= pl.col("base_scr") * 1.5))
            .then(1).otherwise(0).alias("aki_event")
        )
    )

    # 2. ラベルの生成
    future_labels = (
        labels.sort(["hadm_id", "charttime"])
        .rolling(index_column="charttime", period=PRED_WINDOW, group_by="hadm_id")
        .agg(pl.col("aki_event").max().alias("future_aki_label"))
        .with_columns(
            pl.col("future_aki_label").shift(-1).over("hadm_id")
        )
    )

    # 3. 特徴量の生成
    features = (
        hosp_labs_inpatient.sort(["hadm_id", "charttime"])
        .rolling(index_column="charttime", period=T_WINDOW, group_by="hadm_id")
        .agg([
            pl.col("valuenum").mean().alias("feat_mean"),
            pl.col("valuenum").max().alias("feat_max"),
            pl.col("valuenum").std().alias("feat_std"),
            pl.col("valuenum").last().alias("curr_scr")
        ])
    )

    # 4. スライディングアライメント
    features_sorted = features.sort(["hadm_id", "charttime"])
    labels_sorted = future_labels.select(["hadm_id", "charttime", "future_aki_label"]).sort(["hadm_id", "charttime"])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Sortedness of columns cannot be checked.*")
        
        dataset = features_sorted.join_asof(
            labels_sorted,
            on="charttime",
            by="hadm_id",
            strategy="forward"
        ).drop_nulls(subset=["future_aki_label"])

    pts_features = hosp_pts.select(["subject_id", "hadm_id", "age", "gender", "baseline_scr"]).with_columns([
        pl.when(pl.col("gender") == "M").then(1).otherwise(0).alias("gender")
    ])

    dataset = (
        dataset.join(pts_features, on="hadm_id", how="inner")
        .with_columns([
            (pl.col("curr_scr") / pl.col("baseline_scr")).alias("scr_ratio"),
            (pl.col("curr_scr") - pl.col("baseline_scr")).alias("scr_delta")
        ])
    )
    return dataset


def split_and_save_datasets(dataset, output_dir="processed_data", train_ratio=0.7, tune_ratio=0.15, test_ratio=0.15):
    """
    Train, Tune, Test に分割する。
    """
    print(f"データセットの分割 (Train:{train_ratio}, Tune:{tune_ratio}, Test:{test_ratio})...")
    
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
        path = os.path.join(output_dir, f"hosp_{name}.parquet")
        df.write_parquet(path)
        print(f"{name.upper()} セットの保存先: {path} (サンプル数: {len(df)}, 患者数: {df['subject_id'].n_unique()})")

    return train_data, tune_data, test_data


if __name__ == "__main__":
    h_pts, h_labs = process_hosp_with_checks()
    h_dataset = generate_hosp_dataset(h_pts, h_labs)
    print(f"\nHOSP データセットのサンプル数: {len(h_dataset)}")
    print(f"AKI ラベルの分布: \n{h_dataset['future_aki_label'].value_counts()}")

    train_df, tune_df, test_df = split_and_save_datasets(h_dataset)
    print("Done.")