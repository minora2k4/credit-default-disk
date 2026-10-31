"""Gold features -> điểm vỡ nợ bằng mô hình ONNX, và GATE: ROC-AUC tập test phải bằng giá trị của notebook.

Gate chứng minh cả hai điều cùng lúc: đặc trưng Spark khớp notebook VÀ cách chia tập test khớp notebook.
Chạy: make score
"""
import sys

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from credit_pipeline.config import GOLD, load_model_metadata
from credit_pipeline.scoring import add_scores

AUC_TOLERANCE = 1e-5
FEATURES_PATH = f"{GOLD}/features_cash"
PREDICTIONS_PATH = f"{GOLD}/predictions_batch"
IDENTIFIERS = ["SK_ID_CURR", "TARGET", "split"]


def main():
    spark = SparkSession.builder.appName("batch_score").getOrCreate()
    metadata = load_model_metadata(spark)
    feature_names = metadata["feature_names"]
    threshold = metadata["decision_threshold"]

    features = spark.read.parquet(FEATURES_PATH)
    scored = add_scores(spark, features, feature_names, threshold, IDENTIFIERS)
    scored.coalesce(4).write.mode("overwrite").parquet(PREDICTIONS_PATH)
    print(f"Đã ghi {PREDICTIONS_PATH}", flush=True)

    predictions = spark.read.parquet(PREDICTIONS_PATH)
    print(f"Số dòng: {predictions.count():,} | quyết định: "
          f"{ {r['decision']: r['count'] for r in predictions.groupBy('decision').count().collect()} }")

    # --- Chỉ số trên tập test hold-out ---
    test = predictions.filter(F.col("split") == "test").select("TARGET", "probability", "decision").toPandas()
    y_true = test["TARGET"].to_numpy()
    probabilities = test["probability"].to_numpy()
    y_pred = (probabilities >= threshold).astype(int)

    auc = roc_auc_score(y_true, probabilities)
    expected_onnx_auc = metadata["onnx_roc_auc_test"]      # AUC của chính file ONNX, notebook đã tính lúc export
    native_auc = metadata["metrics_test"]["roc_auc"]       # AUC của LightGBM gốc (float64)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    expected = metadata["metrics_test"]

    print(f"\nTập test: {len(test):,} hồ sơ, tỉ lệ vỡ nợ {y_true.mean():.4f}, ngưỡng {threshold:.6f}")
    print(f"ROC-AUC        Spark {auc:.7f} | notebook (ONNX) {expected_onnx_auc:.7f} | notebook (LightGBM gốc) {native_auc:.7f}")
    print(f"Precision lớp 1: Spark {precision[0]:.4f} | notebook {expected['precision']:.4f}")
    print(f"Recall lớp 1   : Spark {recall[0]:.4f} | notebook {expected['recall']:.4f}")
    print(f"F1 lớp 1       : Spark {f1[0]:.4f} | notebook {expected['f1']:.4f}")
    print(f"Confusion matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}   (notebook: TN=45,330 FP=5,670 FN=2,576 TP=2,068)")

    difference = abs(auc - expected_onnx_auc)
    print(f"\n|AUC Spark - AUC ONNX của notebook| = {difference:.2e}  (cho phép {AUC_TOLERANCE:.0e})")
    spark.stop()
    if difference > AUC_TOLERANCE:
        print("GATE CHƯA ĐẠT: đặc trưng hoặc tập test chưa khớp notebook")
        sys.exit(1)
    print("GATE ĐẠT: pipeline Spark tái tạo đúng kết quả của notebook")


if __name__ == "__main__":
    main()
