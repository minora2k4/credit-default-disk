"""Sinh hồ sơ vay TỔNG HỢP theo phân phối của toàn bộ hồ sơ Cash loans trong application_train.csv.

Cách sinh:
  1. Bảng application: SDV GaussianCopula, fit RIÊNG cho từng lớp TARGET (0 / 1).
     Fit chung một mô hình thì quan hệ "điểm EXT_SOURCE thấp -> dễ vỡ nợ" gần như mất hết (đã đo:
     corr(EXT_SOURCE_3, TARGET) từ -0.18 xuống -0.05); fit riêng từng lớp thì giữ lại được.
  2. Bảng bureau + bureau_balance: mỗi hồ sơ sinh được mượn khối CIC của một hồ sơ thật CÙNG LỚP TARGET,
     đổi ID khoản vay và nhiễu số tiền +-10% (cùng một hệ số cho mọi cột tiền của một khoản vay,
     nên tỉ lệ dư nợ / hạn mức không đổi). Đây không phải SDV: SDV nhiều bảng quá nặng với 1.3 triệu khoản vay.

Giới hạn cần nhớ: dữ liệu sinh để thử TẢI và THỬ LUỒNG, không để ước lượng chất lượng mô hình. Bộ sinh học từ
toàn bộ dữ liệu (kể cả phần notebook dùng để train), nên AUC đo trên dữ liệu sinh không thay thế AUC test của notebook.

Đầu ra (trong --out-dir/sim_pool/synthetic):
  applications.jsonl.gz   cùng định dạng message với build_pool.py
  outcomes.parquet        application_id, TARGET
  fidelity_report.txt     so sánh phân phối dữ liệu sinh với dữ liệu thật
"""
import argparse
import gzip
import io
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.metrics import roc_auc_score

from credit_pipeline.columns import APPLICATION_COLUMNS, BUREAU_AMOUNT_COLUMNS
from credit_pipeline.holdout import REQUIRED_AMOUNT_COLUMNS
from pool_common import MAX_MESSAGE_BYTES, balance_by_loan, read_balance_for_loans, records_without_nan

warnings.filterwarnings("ignore")

SYNTHETIC_ID_OFFSET = 900_000_000       # SK_ID_CURR sinh ra nằm ngoài dải ID thật (100002 - 456255)
SYNTHETIC_LOAN_ID_OFFSET = 90_000_000   # SK_ID_BUREAU sinh ra nằm ngoài dải ID thật, vẫn vừa int32
RETIREE_EMPLOYED_CODE = 365243          # DAYS_EMPLOYED của người hưu trí (notebook cũng coi là NaN)
LOW_CARDINALITY_LIMIT = 500             # cột số có ít giá trị khác nhau hơn mức này được coi là phân loại
AMOUNT_NOISE = 0.10
KS_SAMPLE_ROWS = 50_000
KEY_COLUMNS_FOR_TARGET_CORRELATION = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_CREDIT", "AMT_ANNUITY",
]


def load_real_cash_applications(data_dir):
    application = pd.read_csv(os.path.join(data_dir, "application_train.csv"), usecols=APPLICATION_COLUMNS)
    application = application.dropna(subset=REQUIRED_AMOUNT_COLUMNS)
    return application[application["NAME_CONTRACT_TYPE"] == "Cash loans"].reset_index(drop=True)


def find_categorical_columns(features):
    """Cột cờ, cột đếm và mọi cột số ít giá trị khác nhau. SDV tự đoán sai các cột cờ 0/1 có hậu tố _CITY thành 'city'
    (sinh tên thành phố giả), và dùng phân phối liên tục cho cột đếm thì làm lệch phân phối (đã đo bằng KS)."""
    explicit = [
        column for column in features.columns
        if column.startswith(("REG_", "LIVE_", "FLAG_")) and column not in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY")
    ] + ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "HOUR_APPR_PROCESS_START", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
    low_cardinality = [
        column for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column]) and features[column].nunique() <= LOW_CARDINALITY_LIMIT
    ]
    return sorted(set(explicit) | set(low_cardinality))


def build_metadata(features, categorical_columns):
    metadata = Metadata.detect_from_dataframes({"application": features})
    for column in categorical_columns:
        metadata.update_column(column, table_name="application", sdtype="categorical")
    return metadata


def synthesize_applications(real, num_rows, seed, continuous_distribution="beta"):
    """SDV GaussianCopula fit riêng từng lớp TARGET; trả về DataFrame đã xáo trộn, có cột TARGET"""
    feature_columns = [c for c in APPLICATION_COLUMNS if c not in ("SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE")]
    features = real[feature_columns].copy()
    # Mã hưu trí 365243 nằm quá xa phần còn lại của DAYS_EMPLOYED, làm SDV dồn cả phần giá trị thật về cận dưới.
    # Đổi thành NaN để SDV mô hình hóa "thiếu" như một biến riêng, sinh xong thì trả lại mã cũ.
    features.loc[features["DAYS_EMPLOYED"] == RETIREE_EMPLOYED_CODE, "DAYS_EMPLOYED"] = np.nan
    categorical_columns = find_categorical_columns(features)
    continuous_columns = [c for c in features.columns if c not in categorical_columns and pd.api.types.is_numeric_dtype(features[c])]
    distributions = {c: continuous_distribution for c in continuous_columns}

    default_rate = real["TARGET"].mean()
    parts = []
    started = time.time()
    for label in (0, 1):
        subset = features.loc[(real["TARGET"] == label).to_numpy()].reset_index(drop=True)
        synthesizer = GaussianCopulaSynthesizer(build_metadata(subset, categorical_columns), numerical_distributions=distributions)
        synthesizer.fit({"application": subset})
        class_rows = int(round(num_rows * (default_rate if label == 1 else 1 - default_rate)))
        sample = synthesizer.sample(num_rows=class_rows)
        sample = sample["application"] if isinstance(sample, dict) else sample
        sample["DAYS_EMPLOYED"] = sample["DAYS_EMPLOYED"].fillna(RETIREE_EMPLOYED_CODE).round().astype("int64")
        sample["TARGET"] = label
        parts.append(sample[feature_columns + ["TARGET"]])
        print(f"   lớp TARGET={label}: fit trên {len(subset)} dòng, sinh {class_rows} dòng ({time.time() - started:.0f}s)")
    synthetic = pd.concat(parts, ignore_index=True)
    return synthetic.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def fidelity_report(real, synthetic, real_bureau_stats, synthetic_bureau_stats):
    buffer = io.StringIO()

    def say(text=""):
        print(text)
        buffer.write(text + "\n")

    feature_columns = [c for c in synthetic.columns if c not in ("TARGET", "SK_ID_CURR", "NAME_CONTRACT_TYPE")]
    say("== Độ giống phân phối: dữ liệu sinh so với toàn bộ Cash loans thật ==")
    say(f"Số dòng: thật {len(real)} | sinh {len(synthetic)}")
    say(f"Tỉ lệ vỡ nợ: thật {real['TARGET'].mean():.4f} | sinh {synthetic['TARGET'].mean():.4f}")

    missing_gap = (real[feature_columns].isna().mean() - synthetic[feature_columns].isna().mean()).abs()
    say(f"Tỉ lệ thiếu giá trị: lệch lớn nhất {missing_gap.max():.4f} (cột {missing_gap.idxmax()}) | lệch trung bình {missing_gap.mean():.4f}")

    real_sample = real.sample(min(KS_SAMPLE_ROWS, len(real)), random_state=0)
    synthetic_sample = synthetic.sample(min(KS_SAMPLE_ROWS, len(synthetic)), random_state=0)
    ks, tv = {}, {}
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(real[column]):
            a, b = real_sample[column].dropna(), synthetic_sample[column].dropna().astype(float)
            ks[column] = ks_2samp(a, b).statistic
        else:
            a = real[column].fillna("<thiếu>").value_counts(normalize=True)
            b = synthetic[column].fillna("<thiếu>").value_counts(normalize=True)
            tv[column] = 0.5 * a.sub(b, fill_value=0).abs().sum()
    ks, tv = pd.Series(ks), pd.Series(tv)
    say(f"Cột số ({len(ks)}): KS trung bình {ks.mean():.4f} | lớn nhất {ks.max():.4f} (cột {ks.idxmax()})   [0 = giống hệt]")
    say("   5 cột lệch nhất: " + ", ".join(f"{name} {value:.3f}" for name, value in ks.sort_values(ascending=False).head(5).items()))
    say(f"Cột phân loại ({len(tv)}): TV trung bình {tv.mean():.4f} | lớn nhất {tv.max():.4f} (cột {tv.idxmax()})   [0 = giống hệt]")
    say(f"DAYS_EMPLOYED = 365243 (mã người hưu trí): thật {(real['DAYS_EMPLOYED'] == 365243).mean():.3f} | sinh {(synthetic['DAYS_EMPLOYED'] == 365243).mean():.3f}")

    say()
    say("== Quan hệ với TARGET (điều quyết định mô hình có tín hiệu để dự đoán) ==")
    for column in KEY_COLUMNS_FOR_TARGET_CORRELATION:
        r = real[[column, "TARGET"]].astype(float).corr().iloc[0, 1]
        s = synthetic[[column, "TARGET"]].astype(float).corr().iloc[0, 1]
        say(f"corr({column:14s}, TARGET): thật {r:+.4f} | sinh {s:+.4f}")
    for name, frame in (("thật", real), ("sinh", synthetic)):
        score = frame[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype(float).mean(axis=1)
        known = score.notna()
        say(f"AUC của riêng EXT_SOURCE trung bình ({name}): {roc_auc_score(frame.loc[known, 'TARGET'], -score[known]):.4f}")

    say()
    say("== Bureau (mượn từ hồ sơ thật cùng lớp, nên chỉ để đối chiếu) ==")
    for key in ("tỉ lệ hồ sơ có CIC", "số khoản CIC TB mỗi hồ sơ có CIC", "số dòng balance TB mỗi khoản"):
        say(f"{key}: thật {real_bureau_stats[key]:.3f} | sinh {synthetic_bureau_stats[key]:.3f}")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--out-dir", default="work")
    parser.add_argument("--num-rows", type=int, default=55_644, help="mặc định bằng cỡ tập test để dễ so sánh")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pool_dir = os.path.join(args.out_dir, "sim_pool", "synthetic")
    os.makedirs(pool_dir, exist_ok=True)

    # 1. Toàn bộ Cash loans thật (cùng điều kiện lọc với notebook)
    real = load_real_cash_applications(args.data_dir)
    print(f"1. dữ liệu thật: {len(real)} hồ sơ Cash | tỉ lệ vỡ nợ {real['TARGET'].mean():.4f}")

    # 2. Sinh bảng application
    print("2. sinh bảng application bằng SDV GaussianCopula (fit riêng từng lớp TARGET):")
    synthetic = synthesize_applications(real, args.num_rows, args.seed)
    synthetic["SK_ID_CURR"] = SYNTHETIC_ID_OFFSET + np.arange(len(synthetic), dtype="int64")
    synthetic["NAME_CONTRACT_TYPE"] = "Cash loans"

    # 3. Chọn hồ sơ thật cùng lớp để mượn khối CIC
    donor_ids = np.empty(len(synthetic), dtype="int64")
    for label in (0, 1):
        mask = (synthetic["TARGET"] == label).to_numpy()
        donor_ids[mask] = rng.choice(real.loc[real["TARGET"] == label, "SK_ID_CURR"].to_numpy(), size=int(mask.sum()), replace=True)
    donor_id_set = set(donor_ids.tolist())

    bureau_all = pd.read_csv(os.path.join(args.data_dir, "bureau.csv"))
    real_ids = set(real["SK_ID_CURR"])
    real_bureau = bureau_all[bureau_all["SK_ID_CURR"].isin(real_ids)]
    donor_bureau = real_bureau[real_bureau["SK_ID_CURR"].isin(donor_id_set)]
    balance = read_balance_for_loans(os.path.join(args.data_dir, "bureau_balance.csv"), set(donor_bureau["SK_ID_BUREAU"]))
    balance_by_loan_id = balance_by_loan(balance)
    print(f"3. mượn CIC từ {len(donor_id_set)} hồ sơ thật: {len(donor_bureau)} khoản vay, {len(balance)} dòng balance")
    del balance
    loans_by_donor = {
        donor: records_without_nan(group.drop(columns=["SK_ID_CURR"]))
        for donor, group in donor_bureau.groupby("SK_ID_CURR")
    }

    # 4. Ghép message
    next_loan_id = SYNTHETIC_LOAN_ID_OFFSET
    sizes, loan_counts, balance_counts = [], [], []
    application_columns = [c for c in APPLICATION_COLUMNS if c != "TARGET"]
    records = records_without_nan(synthetic[application_columns])
    out_path = os.path.join(pool_dir, "applications.jsonl.gz")
    with gzip.open(out_path, "wt", encoding="utf-8") as out_file:
        for record, donor_id in zip(records, donor_ids.tolist()):
            loans = []
            for donor_loan in loans_by_donor.get(donor_id, []):
                loan = dict(donor_loan)
                factor = float(rng.uniform(1 - AMOUNT_NOISE, 1 + AMOUNT_NOISE))
                for column in BUREAU_AMOUNT_COLUMNS:
                    if loan[column] is not None:
                        loan[column] = round(loan[column] * factor, 3)
                rows = balance_by_loan_id.get(donor_loan["SK_ID_BUREAU"], [])
                loan["SK_ID_BUREAU"] = next_loan_id
                next_loan_id += 1
                loan["balance"] = [{"MONTHS_BALANCE": month, "STATUS": status} for month, status in rows]
                balance_counts.append(len(rows))
                loans.append(loan)
            loan_counts.append(len(loans))
            message = {
                "schema_version": 1,
                "application_id": str(record["SK_ID_CURR"]),
                "application": record,
                "bureau": loans,
            }
            line = json.dumps(message, allow_nan=False, separators=(",", ":"))
            sizes.append(len(line.encode("utf-8")))
            out_file.write(line + "\n")
    assert max(sizes) < MAX_MESSAGE_BYTES, "Có message vượt 4 MB, phải nâng giới hạn của topic"
    print(f"4. ghi {len(sizes)} message vào {out_path} | kích thước TB {sum(sizes) / len(sizes) / 1024:.1f} KB, lớn nhất {max(sizes) / 1024:.1f} KB")

    outcomes = pd.DataFrame({"application_id": synthetic["SK_ID_CURR"].astype(str), "TARGET": synthetic["TARGET"].astype(int)})
    outcomes.to_parquet(os.path.join(pool_dir, "outcomes.parquet"), index=False)

    # 5. Báo cáo độ giống phân phối
    real_loans_per_app = real_bureau.groupby("SK_ID_CURR").size()
    real_balance_rows = read_balance_for_loans(os.path.join(args.data_dir, "bureau_balance.csv"), set(real_bureau["SK_ID_BUREAU"]))
    real_stats = {
        "tỉ lệ hồ sơ có CIC": len(real_loans_per_app) / len(real),
        "số khoản CIC TB mỗi hồ sơ có CIC": real_loans_per_app.mean(),
        "số dòng balance TB mỗi khoản": len(real_balance_rows) / len(real_bureau),
    }
    del real_balance_rows
    synthetic_stats = {
        "tỉ lệ hồ sơ có CIC": float(np.mean(np.array(loan_counts) > 0)),
        "số khoản CIC TB mỗi hồ sơ có CIC": float(np.mean([c for c in loan_counts if c > 0])),
        "số dòng balance TB mỗi khoản": float(np.mean(balance_counts)),
    }
    print("5. báo cáo độ giống phân phối:\n")
    report = fidelity_report(real, synthetic, real_stats, synthetic_stats)
    with open(os.path.join(pool_dir, "fidelity_report.txt"), "w", encoding="utf-8") as report_file:
        report_file.write(report)


if __name__ == "__main__":
    main()
