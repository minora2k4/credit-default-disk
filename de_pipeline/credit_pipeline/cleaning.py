"""Làm sạch dữ liệu thô (bronze -> silver), theo main.ipynb mục 1-3.

Batch và streaming dùng CHUNG các hàm này, để một hồ sơ được xử lý giống hệt nhau dù đến từ file CSV hay từ Kafka.
Các hàm nhận Spark DataFrame và trả Spark DataFrame; không đọc/ghi file.
"""
from pyspark.sql import functions as F

from credit_pipeline.columns import APPLICATION_COLUMNS, REQUIRED_AMOUNT_COLUMNS

RETIREE_EMPLOYED_CODE = 365243   # DAYS_EMPLOYED của người hưu trí (~1000 năm), không phải số ngày đi làm thật


def clean_application(application):
    """Notebook cell 8-9. Khác notebook ở một điểm: KHÔNG xóa hồ sơ thiếu số tiền mà gắn cờ HAS_REQUIRED_AMOUNTS.

    Notebook dropna 4 cột AMT; ở đây giữ dòng để streaming vẫn ghi nhận được hồ sơ lỗi (status INVALID_INPUT).
    Bước tạo đặc trưng lọc theo cờ này, nên tập hồ sơ được mô hình hóa vẫn đúng bằng notebook.
    """
    # Hồ sơ gửi qua Kafka không có TARGET, nên chỉ lấy các cột có mặt
    columns = [name for name in APPLICATION_COLUMNS if name in application.columns]
    has_required_amounts = F.lit(True)
    for name in REQUIRED_AMOUNT_COLUMNS:
        has_required_amounts = has_required_amounts & F.col(name).isNotNull()

    return (
        application.select(*columns)
        .withColumn("HAS_REQUIRED_AMOUNTS", has_required_amounts)
        # Mã hưu trí -> null (thông tin hưu trí vẫn còn ở NAME_INCOME_TYPE). pandas biến cột thành float khi có NaN,
        # nên ép double ở mọi trường hợp để kiểu giống notebook
        .withColumn(
            "DAYS_EMPLOYED",
            F.when(F.col("DAYS_EMPLOYED") == RETIREE_EMPLOYED_CODE, F.lit(None).cast("double"))
            .otherwise(F.col("DAYS_EMPLOYED").cast("double")),
        )
        # Thiếu nghề nghiệp phần lớn là người hưu trí, không phải thất nghiệp
        .withColumn("OCCUPATION_TYPE", F.coalesce(F.col("OCCUPATION_TYPE"), F.lit("Not declared")))
    )


def clean_bureau(bureau):
    """Notebook cell 23: chỉ chuyển giá trị sai logic thành null, không xóa dòng, không điền median"""
    no_value = F.lit(None).cast("double")
    return (
        bureau
        # Ngày kết thúc (dự kiến / thực tế) nằm trước ngày mở khoản vay. So sánh với null cho kết quả null nên giữ nguyên giá trị
        .withColumn("DAYS_CREDIT_ENDDATE",
                    F.when(F.col("DAYS_CREDIT_ENDDATE") < F.col("DAYS_CREDIT"), no_value).otherwise(F.col("DAYS_CREDIT_ENDDATE")))
        .withColumn("DAYS_ENDDATE_FACT",
                    F.when(F.col("DAYS_ENDDATE_FACT") < F.col("DAYS_CREDIT"), no_value).otherwise(F.col("DAYS_ENDDATE_FACT")))
        # Dư nợ âm -> 0 (giống clip(lower=0): null vẫn là null)
        .withColumn("AMT_CREDIT_SUM_DEBT",
                    F.when(F.col("AMT_CREDIT_SUM_DEBT") < 0, F.lit(0.0)).otherwise(F.col("AMT_CREDIT_SUM_DEBT")))
        # Hạn mức thẻ <= 0 -> không xác định
        .withColumn("AMT_CREDIT_SUM_LIMIT",
                    F.when(F.col("AMT_CREDIT_SUM_LIMIT") <= 0, no_value).otherwise(F.col("AMT_CREDIT_SUM_LIMIT")))
    )


def clean_bureau_balance(bureau_balance):
    """Notebook cell 36 (phần làm sạch): STATUS -> mức quá hạn DPD (0..5; C và X -> null) và hai cờ IS_X, IS_C.

    Các phép gộp theo cửa sổ 12/24 tháng làm ở bước tạo đặc trưng (features.py), không phải ở đây.
    """
    status = F.col("STATUS")
    return (
        bureau_balance.select("SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS")
        # '0'..'5' -> số; 'C' (đã đóng) và 'X' (không rõ) -> null, giống pd.to_numeric(errors="coerce")
        .withColumn("DPD", F.when(status.isin("0", "1", "2", "3", "4", "5"), status.cast("double")))
        .withColumn("IS_X", F.when(status == "X", 1).otherwise(0).cast("tinyint"))
        .withColumn("IS_C", F.when(status == "C", 1).otherwise(0).cast("tinyint"))
    )
