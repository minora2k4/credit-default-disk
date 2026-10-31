"""Chấm điểm bằng mô hình ONNX trong Spark: mapInPandas + onnxruntime. Batch và streaming dùng CHUNG hàm này.

Vì sao không viết bằng DataFrame API: mô hình là một file ONNX, chỉ chạy được bằng onnxruntime của Python.
mapInPandas chuyển dữ liệu từng lô (mặc định 10.000 dòng) từ JVM sang Python bằng Arrow, ta chấm điểm cả lô một lần
rồi trả về DataFrame kết quả.
"""
import os

import numpy as np
from pyspark import SparkFiles
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from credit_pipeline.config import MODEL_ONNX_PATH, MODEL_VERSION

# Mỗi tiến trình Python worker giữ một session; Spark tái dùng worker giữa các task nên model chỉ nạp một lần.
# Chỉ được truy cập qua _get_session (hàm nằm trong module, được pickle theo tham chiếu), nếu không dict sẽ bị
# sao chép theo từng closure và mất tác dụng cache.
_SESSIONS = {}


def _get_session(model_path):
    session = _SESSIONS.get(model_path)
    if session is None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        # Mỗi task một luồng: Spark đã chạy song song nhiều task, nhiều luồng trong mỗi task chỉ tranh CPU nhau
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        _SESSIONS[model_path] = session
    return session


def _probabilities_of_default(outputs):
    """Giống predict_onnx_probabilities trong notebook: outputs[1] là xác suất của từng lớp, lấy cột lớp 1 (vỡ nợ)"""
    probabilities = outputs[1]
    if isinstance(probabilities, list):   # định dạng zipmap: mỗi dòng là một dict {lớp: xác suất}
        probabilities = np.array([row[1] for row in probabilities])
    return np.asarray(probabilities)[:, 1]


def add_scores(spark, features, feature_names, threshold, passthrough_columns, model_version=MODEL_VERSION):
    """Thêm cột probability (xác suất vỡ nợ), decision (REJECT nếu >= ngưỡng, ngược lại APPROVE) và model_version.

    features: DataFrame có đủ feature_names; passthrough_columns: các cột định danh giữ lại ở kết quả.
    """
    # addFile tải file mô hình từ HDFS về thư mục cục bộ của từng executor. Đường dẫn phải đủ scheme hdfs://,
    # nếu không Spark hiểu là file cục bộ
    default_fs = spark.conf.get("spark.hadoop.fs.defaultFS")
    spark.sparkContext.addFile(default_fs + MODEL_ONNX_PATH)
    model_file = os.path.basename(MODEL_ONNX_PATH)

    schema = StructType(
        [features.schema[name] for name in passthrough_columns]
        + [StructField("probability", DoubleType()), StructField("decision", StringType()),
           StructField("model_version", StringType())]
    )
    feature_names = list(feature_names)
    passthrough_columns = list(passthrough_columns)

    def score_batches(batches):
        session = _get_session(SparkFiles.get(model_file))
        input_name = session.get_inputs()[0].name
        for frame in batches:
            if frame.empty:
                continue
            # float32 và giữ NaN: giống hệt X_test_float trong notebook (mục 6)
            matrix = frame[feature_names].to_numpy(dtype=np.float32)
            probabilities = _probabilities_of_default(session.run(None, {input_name: matrix}))
            result = frame[passthrough_columns].copy()
            result["probability"] = probabilities.astype("float64")
            result["decision"] = np.where(probabilities >= threshold, "REJECT", "APPROVE")
            result["model_version"] = model_version
            yield result

    return features.mapInPandas(score_batches, schema)
