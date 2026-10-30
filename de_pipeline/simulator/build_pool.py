"""Tạo sim pool: mỗi hồ sơ trong tập test là một JSON lồng nhau (hồ sơ + các khoản CIC + lịch sử tháng).

Dữ liệu giữ nguyên dạng THÔ như trong CSV (chưa làm sạch, chưa tạo đặc trưng, không có TARGET):
việc làm sạch là của Spark, để batch và streaming dùng chung một đoạn code.

Đây là chế độ "real-test": phát lại 55,644 hồ sơ THẬT của tập test để chứng minh pipeline tái tạo đúng kết quả
của notebook (AUC 0.7801). Chế độ sinh dữ liệu mới nằm ở synthesize_pool.py.

Đầu ra (trong --out-dir):
  holdout_ids.parquet                      SK_ID_CURR, TARGET, split của toàn bộ 278,220 hồ sơ Cash
  sim_pool/real_test/applications.jsonl.gz   một dòng = một message gửi vào Kafka (55,644 dòng)
  sim_pool/real_test/outcomes.parquet        application_id, TARGET của tập test (nhãn gửi trễ)
"""
import argparse
import gzip
import json
import os

import pandas as pd

from credit_pipeline.holdout import build_holdout_ids
from pool_common import MAX_MESSAGE_BYTES, balance_by_loan, read_balance_for_loans, records_without_nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--out-dir", default="work")
    args = parser.parse_args()
    pool_dir = os.path.join(args.out_dir, "sim_pool", "real_test")
    os.makedirs(pool_dir, exist_ok=True)

    # 1. Tập test của notebook
    holdout_ids = build_holdout_ids(os.path.join(args.data_dir, "application_train.csv"))
    holdout_ids.to_parquet(os.path.join(args.out_dir, "holdout_ids.parquet"), index=False)
    test_ids = holdout_ids.loc[holdout_ids["split"] == "test", "SK_ID_CURR"].tolist()
    test_id_set = set(test_ids)
    print(f"1. holdout: {len(holdout_ids)} hồ sơ Cash | train {(holdout_ids['split'] == 'train').sum()} | "
          f"test {len(test_ids)} | tỉ lệ vỡ nợ test {holdout_ids.loc[holdout_ids['split'] == 'test', 'TARGET'].mean():.4f}")

    # 2. Hồ sơ thô của tập test (giữ thứ tự của tập test), bỏ TARGET
    application = pd.read_csv(os.path.join(args.data_dir, "application_train.csv"))
    application = application[application["SK_ID_CURR"].isin(test_id_set)].set_index("SK_ID_CURR", drop=False)
    application = application.loc[test_ids].drop(columns=["TARGET"])
    print(f"2. application: {application.shape}")

    # 3. Các khoản CIC của tập test
    bureau = pd.read_csv(os.path.join(args.data_dir, "bureau.csv"))
    bureau = bureau[bureau["SK_ID_CURR"].isin(test_id_set)]
    bureau_ids = set(bureau["SK_ID_BUREAU"])
    print(f"3. bureau: {len(bureau)} khoản vay của {bureau['SK_ID_CURR'].nunique()} hồ sơ")

    # 4. Lịch sử tháng: đọc theo chunk (27 triệu dòng), chỉ giữ các khoản vay liên quan để tiết kiệm RAM
    balance = read_balance_for_loans(os.path.join(args.data_dir, "bureau_balance.csv"), bureau_ids)
    print(f"4. bureau_balance: {len(balance)} dòng cho {balance['SK_ID_BUREAU'].nunique()} khoản vay")
    balance_by_loan_id = balance_by_loan(balance)
    del balance

    # 5. Ghép mỗi hồ sơ thành một message JSON lồng nhau
    bureau_by_application = {}
    for application_id, group in bureau.groupby("SK_ID_CURR"):
        loans = records_without_nan(group.drop(columns=["SK_ID_CURR"]))
        for loan in loans:
            loan["balance"] = [{"MONTHS_BALANCE": m, "STATUS": st} for m, st in balance_by_loan_id.get(loan["SK_ID_BUREAU"], [])]
        bureau_by_application[application_id] = loans

    sizes = []
    max_loans = 0
    out_path = os.path.join(pool_dir, "applications.jsonl.gz")
    with gzip.open(out_path, "wt", encoding="utf-8") as out_file:
        for record in records_without_nan(application):
            loans = bureau_by_application.get(record["SK_ID_CURR"], [])
            message = {
                "schema_version": 1,
                "application_id": str(record["SK_ID_CURR"]),
                "application": record,
                "bureau": loans,
            }
            line = json.dumps(message, allow_nan=False, separators=(",", ":"))
            sizes.append(len(line.encode("utf-8")))
            max_loans = max(max_loans, len(loans))
            out_file.write(line + "\n")
    print(f"5. ghi {len(sizes)} message vào {out_path}")
    print(f"   kích thước message: trung bình {sum(sizes) / len(sizes) / 1024:.1f} KB | lớn nhất {max(sizes) / 1024:.1f} KB "
          f"| nhiều khoản CIC nhất: {max_loans}")
    assert max(sizes) < MAX_MESSAGE_BYTES, "Có message vượt 4 MB, phải nâng giới hạn của topic"

    # 6. Nhãn của tập test, producer sẽ gửi trễ vào topic loan_outcomes
    outcomes = holdout_ids.loc[holdout_ids["split"] == "test", ["SK_ID_CURR", "TARGET"]].copy()
    outcomes["application_id"] = outcomes["SK_ID_CURR"].astype(str)
    outcomes[["application_id", "TARGET"]].to_parquet(os.path.join(pool_dir, "outcomes.parquet"), index=False)
    print(f"6. outcomes: {len(outcomes)} dòng")


if __name__ == "__main__":
    main()
