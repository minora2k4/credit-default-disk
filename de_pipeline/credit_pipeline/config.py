"""Đường dẫn HDFS dùng chung giữa các job."""
import json

DATALAKE = "/datalake"
SILVER = f"{DATALAKE}/silver"
GOLD = f"{DATALAKE}/gold"
REFERENCE_NOTEBOOK = f"{DATALAKE}/reference/notebook"
REFERENCE_HOLDOUT = f"{DATALAKE}/reference/holdout_ids"

MODEL_DIR = "/models/credit_default/lightgbm/v1"
MODEL_METADATA_PATH = f"{MODEL_DIR}/metadata.json"
MODEL_ONNX_PATH = f"{MODEL_DIR}/model.onnx"
MODEL_VERSION = "lightgbm-v1"


def load_model_metadata(spark, path=MODEL_METADATA_PATH):
    """Đọc metadata.json của mô hình từ HDFS: danh sách 221 đặc trưng (đúng thứ tự mô hình cần) và ngưỡng quyết định"""
    return json.loads("\n".join(spark.sparkContext.textFile(path).collect()))
