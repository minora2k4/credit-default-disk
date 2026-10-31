"""So sánh đặc trưng do Spark tạo với bảng tham chiếu xuất từ notebook (bước 3.6), từng cột một.

  spark-submit tools/compare_features.py --stage application

Với mỗi cột, hai giá trị được coi là khớp khi cùng null, hoặc bằng nhau (kể cả inf), hoặc lệch tương đối < --tolerance.
Thoát với mã lỗi khác 0 nếu có bất kỳ cột nào lệch, để dùng được trong Makefile / CI.
"""
import argparse
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from credit_pipeline.config import REFERENCE_NOTEBOOK, SILVER, load_model_metadata
from credit_pipeline.features import (
    application_features, bureau_balance_loan_features, bureau_customer_features, merge_and_fill,
)
from credit_pipeline.naming import sanitize_name

KEY = "SK_ID_CURR"


def compare(ours, reference, key, tolerance, label):
    """In báo cáo so sánh; trả về True nếu mọi cột khớp. Cả hai bảng phải có cột `key` và cùng tên cột (đã sanitize)."""
    ours_columns = [name for name in ours.columns if name != key]
    reference_columns = [name for name in reference.columns if name != key]
    only_ours = sorted(set(ours_columns) - set(reference_columns))
    only_reference = sorted(set(reference_columns) - set(ours_columns))
    common = [name for name in ours_columns if name in set(reference_columns)]

    # Đổi tên cột thành s0, r0, s1, r1... để tránh ký tự đặc biệt (CREDIT/TOTAL, dấu cách...) trong biểu thức
    ours_renamed = ours.select(key, F.lit(1).alias("_in_s"), *[F.col(f"`{name}`").cast("double").alias(f"s{i}") for i, name in enumerate(common)])
    reference_renamed = reference.select(key, F.lit(1).alias("_in_r"), *[F.col(f"`{name}`").cast("double").alias(f"r{i}") for i, name in enumerate(common)])
    joined = ours_renamed.join(reference_renamed, on=key, how="full_outer").cache()

    aggregations = [
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("_in_r").isNull(), 1).otherwise(0)).alias("only_spark"),
        F.sum(F.when(F.col("_in_s").isNull(), 1).otherwise(0)).alias("only_reference"),
    ]
    for i in range(len(common)):
        s, r = F.col(f"s{i}"), F.col(f"r{i}")
        null_differs = s.isNull() != r.isNull()
        both = s.isNotNull() & r.isNotNull()
        relative = F.abs(s - r) / F.greatest(F.lit(1.0), F.abs(s), F.abs(r))
        differs = null_differs | (both & (s != r) & ~(F.isnan(s) & F.isnan(r)) & ~(relative <= tolerance))
        aggregations += [
            F.sum(F.when(null_differs, 1).otherwise(0)).alias(f"n{i}"),
            F.sum(F.when(differs, 1).otherwise(0)).alias(f"d{i}"),
            F.max(F.when(both & (s != r), relative)).alias(f"m{i}"),
        ]
    row = joined.agg(*aggregations).collect()[0]
    joined.unpersist()

    bad = []
    for i, name in enumerate(common):
        if row[f"d{i}"]:
            bad.append((name, row[f"n{i}"], row[f"d{i}"], row[f"m{i}"]))
    largest = max((row[f"m{i}"] or 0.0 for i in range(len(common))), default=0.0)

    print(f"\n===== {label} =====")
    print(f"Số dòng sau khi nối: {row['rows']:,} | chỉ có ở Spark: {row['only_spark']:,} | chỉ có ở tham chiếu: {row['only_reference']:,}")
    print(f"Số cột so sánh: {len(common)} | chỉ có ở Spark: {len(only_ours)} {only_ours[:5]} | chỉ có ở tham chiếu: {len(only_reference)} {only_reference[:5]}")
    print(f"Sai lệch tương đối lớn nhất (trong các ô đều có giá trị): {largest:.3e}")
    if bad:
        print(f"Cột LỆCH ({len(bad)}):  [tên | số ô lệch vị trí null | số ô lệch giá trị | lệch tương đối lớn nhất]")
        for name, nulls, diffs, rel in sorted(bad, key=lambda b: -b[2])[:25]:
            print(f"  {name:45s} null={nulls:>8,} lệch={diffs:>8,} max={rel if rel is not None else float('nan'):.3e}")
    ok = not bad and not only_ours and not only_reference and not row["only_spark"] and not row["only_reference"]
    print("KẾT QUẢ: " + ("ĐẠT, khớp hoàn toàn với notebook" if ok else "CHƯA ĐẠT"))
    return ok


def compare_application(spark, tolerance):
    metadata = load_model_metadata(spark)
    ours = application_features(spark.read.parquet(f"{SILVER}/application"), metadata["feature_names"])
    return compare(ours, read_reference(spark, "application_features"), KEY, tolerance, "3.8a application_features")


def read_reference(spark, name):
    """Bảng tham chiếu chụp trước khi notebook sanitize tên cột (cell 45): sanitize cùng cách để so được"""
    reference = spark.read.parquet(f"{REFERENCE_NOTEBOOK}/{name}.parquet")
    return reference.toDF(*[sanitize_name(column) for column in reference.columns])


def compare_bureau_balance(spark, tolerance):
    ours = bureau_balance_loan_features(spark.read.parquet(f"{SILVER}/bureau_balance"))
    return compare(ours, read_reference(spark, "bureau_balance_features"), "SK_ID_BUREAU", tolerance, "3.8b bureau_balance_features")


def compare_bureau(spark, tolerance):
    bb_loan = bureau_balance_loan_features(spark.read.parquet(f"{SILVER}/bureau_balance"))
    ours = bureau_customer_features(spark.read.parquet(f"{SILVER}/bureau"), bb_loan)
    return compare(ours, read_reference(spark, "bureau_features"), KEY, tolerance, "3.8c bureau_features")


def compare_cash(spark, tolerance):
    names = load_model_metadata(spark)["feature_names"]
    app = application_features(spark.read.parquet(f"{SILVER}/application"), names)
    bb_loan = bureau_balance_loan_features(spark.read.parquet(f"{SILVER}/bureau_balance"))
    bureau = bureau_customer_features(spark.read.parquet(f"{SILVER}/bureau"), bb_loan)
    ours = merge_and_fill(app, bureau, names)
    return compare(ours, read_reference(spark, "application_train_cash"), KEY, tolerance, "3.8d application_train_cash (221 đặc trưng)")


STAGES = {
    "application": compare_application,
    "bureau_balance": compare_bureau_balance,
    "bureau": compare_bureau,
    "cash": compare_cash,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    spark = SparkSession.builder.appName(f"compare_{args.stage}").getOrCreate()
    ok = STAGES[args.stage](spark, args.tolerance)
    spark.stop()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
