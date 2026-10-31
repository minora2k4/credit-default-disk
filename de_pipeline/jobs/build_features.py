"""Silver -> Gold: bảng đặc trưng Cash loans (221 đặc trưng) kèm nhãn và cột split train/test của notebook.

Chạy: make features
"""
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from credit_pipeline.config import GOLD, REFERENCE_HOLDOUT, SILVER, load_model_metadata
from credit_pipeline.features import (
    application_features, bureau_balance_loan_features, bureau_customer_features, merge_and_fill,
)

EXPECTED_ROWS = {"train": 222_576, "test": 55_644}   # số liệu của notebook (mục 5)
OUTPUT_PATH = f"{GOLD}/features_cash"


def main():
    spark = SparkSession.builder.appName("build_features").getOrCreate()
    started = time.time()
    feature_names = load_model_metadata(spark)["feature_names"]

    # 1. Cùng chuỗi hàm với tool so sánh (đã khớp notebook ở bước 3.8)
    application = application_features(spark.read.parquet(f"{SILVER}/application"), feature_names)
    bb_loan = bureau_balance_loan_features(spark.read.parquet(f"{SILVER}/bureau_balance"))
    bureau = bureau_customer_features(spark.read.parquet(f"{SILVER}/bureau"), bb_loan)
    features = merge_and_fill(application, bureau, feature_names)

    # 2. Gắn cột split (train/test) của notebook. holdout_ids do tools/ tái tạo bằng đúng phép chia của notebook
    holdout = spark.read.parquet(REFERENCE_HOLDOUT).select(
        "SK_ID_CURR", "split", F.col("TARGET").alias("HOLDOUT_TARGET"))
    gold = features.join(holdout, on="SK_ID_CURR", how="left")
    gold = gold.select("SK_ID_CURR", "TARGET", "split", *[F.col(f"`{name}`") for name in feature_names])

    # 3. Ghi: ~278 nghìn dòng chỉ vài chục MB, gộp còn 4 file thay vì 24 file nhỏ
    gold.coalesce(4).write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"Đã ghi {OUTPUT_PATH} ({time.time() - started:.0f}s)", flush=True)

    # 4. Đọc lại từ HDFS và kiểm tra
    written = spark.read.parquet(OUTPUT_PATH)
    total = written.count()
    split_counts = {row["split"]: row["count"] for row in written.groupBy("split").count().collect()}
    no_split = split_counts.get(None, 0)
    test_default_rate = written.filter(F.col("split") == "test").agg(F.avg("TARGET")).collect()[0][0]
    # TARGET lấy từ application phải trùng TARGET trong holdout_ids (cùng một hồ sơ, cùng nhãn)
    target_mismatch = (
        features.join(holdout, on="SK_ID_CURR", how="left")
        .filter(F.col("TARGET") != F.col("HOLDOUT_TARGET")).count()
    )

    print(f"Số dòng: {total:,} | số cột: {len(written.columns)} (3 định danh + {len(written.columns) - 3} đặc trưng)")
    print(f"split: {split_counts} | hồ sơ không có split: {no_split}")
    print(f"Tỉ lệ vỡ nợ tập test: {test_default_rate:.4f} | TARGET lệch holdout: {target_mismatch}")
    assert total == sum(EXPECTED_ROWS.values()), f"Cần {sum(EXPECTED_ROWS.values())} dòng, có {total}"
    assert no_split == 0 and target_mismatch == 0
    assert split_counts == EXPECTED_ROWS, f"Số dòng train/test lệch notebook: {split_counts}"
    assert written.columns[3:] == feature_names, "Thứ tự 221 đặc trưng khác metadata.json"
    print("ĐẠT: gold/features_cash khớp notebook (số dòng, split, thứ tự đặc trưng)")
    spark.stop()


if __name__ == "__main__":
    main()
