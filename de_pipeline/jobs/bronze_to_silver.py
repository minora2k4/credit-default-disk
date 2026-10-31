"""Bronze (CSV gốc trên HDFS) -> Silver (Parquet đã làm sạch, có kiểu dữ liệu).

Chạy: make silver
"""
import time

from pyspark.sql import SparkSession

from credit_pipeline.cleaning import clean_application, clean_bureau, clean_bureau_balance
from credit_pipeline.schemas import APPLICATION_SCHEMA, BUREAU_BALANCE_SCHEMA, BUREAU_SCHEMA

BRONZE = "/datalake/bronze/batch"
SILVER = "/datalake/silver"

# (tên thư mục bronze, schema, hàm làm sạch, tên thư mục silver, số dòng đã biết của file CSV gốc)
TABLES = [
    ("application_train", APPLICATION_SCHEMA, clean_application, "application", 307_511),
    ("bureau", BUREAU_SCHEMA, clean_bureau, "bureau", 1_716_428),
    ("bureau_balance", BUREAU_BALANCE_SCHEMA, clean_bureau_balance, "bureau_balance", 27_299_925),
]


def main():
    spark = SparkSession.builder.appName("bronze_to_silver").getOrCreate()
    for bronze_name, schema, clean, silver_name, expected_rows in TABLES:
        started = time.time()
        raw = (
            spark.read.csv(f"{BRONZE}/{bronze_name}/", header=True, schema=schema,
                           # kiểm tra tên cột trong header khớp schema (mặc định Spark chỉ khớp theo vị trí, sai cột cũng không báo)
                           enforceSchema=False)
        )
        silver_path = f"{SILVER}/{silver_name}"
        clean(raw).write.mode("overwrite").parquet(silver_path)

        written_rows = spark.read.parquet(silver_path).count()
        print(f"[{silver_name}] {written_rows:,} dòng ({time.time() - started:.0f}s) -> {silver_path}", flush=True)
        assert written_rows == expected_rows, f"{silver_name}: ghi {written_rows} dòng, CSV gốc có {expected_rows}"
    spark.stop()


if __name__ == "__main__":
    main()
