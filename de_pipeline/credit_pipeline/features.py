"""Tạo đặc trưng bằng Spark, chép lại logic pandas trong main.ipynb. Batch và streaming dùng CHUNG các hàm này.

Mỗi hàm ghi rõ cell của notebook mà nó chép, và những chỗ Spark khác pandas phải xử lý riêng.
"""
from pyspark.sql import functions as F

from credit_pipeline.naming import (
    BB_LOAN_COLUMNS, BUREAU_CUSTOMER_COLUMNS, CONTRACT_DUMMIES, NUMERIC_STATS_COLUMNS, ONE_HOT_SOURCES, SANITIZE_PATTERN,
    one_hot_specs,
)

EXTERNAL_SCORES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def pandas_divide(numerator, denominator):
    """Chia giống pandas. Spark 4 bật ANSI nên x/0 báo lỗi, còn pandas cho: x/0 = +-inf, 0/0 = NaN.

    NaN của pandas được to_parquet lưu thành null, nên 0/0 ở đây trả null để khớp với dữ liệu tham chiếu.
    Chia cho null cho null (cả hai bên giống nhau).
    """
    infinity = F.lit(float("inf"))
    return (
        F.when(denominator == 0, F.when(numerator > 0, infinity).when(numerator < 0, -infinity))
        .otherwise(F.try_divide(numerator, denominator))
    )


def application_features(application, one_hot_names):
    """main.ipynb cell 9 (+ cell 19): tỉ lệ, tổng hợp EXT_SOURCE, one-hot. Nhận bảng đã qua clean_application.

    one_hot_names: danh sách tên đặc trưng của mô hình (metadata.json); hàm tự chọn ra những cột dạng one-hot.
    Chỉ giữ hồ sơ đủ 4 cột số tiền (notebook dropna), nhưng KHÔNG lọc Cash loans ở đây: việc tách Cash/Revolving
    làm ở bước ghép cuối (merge_and_fill), giống cell 41.
    """
    c = F.col
    frame = application.filter(c("HAS_REQUIRED_AMOUNTS")).drop("HAS_REQUIRED_AMOUNTS")

    # 3. Các trường tỉ lệ. PTI, TERM... chia cho cột có thể bằng 0 nên dùng pandas_divide
    frame = (
        frame
        .withColumn("PTI", pandas_divide(c("AMT_ANNUITY"), c("AMT_INCOME_TOTAL")))
        .withColumn("CREDIT/TOTAL", pandas_divide(c("AMT_CREDIT"), c("AMT_INCOME_TOTAL")))
        .withColumn("LTV", pandas_divide(c("AMT_CREDIT"), c("AMT_GOODS_PRICE")))
        .withColumn("TERM", pandas_divide(c("AMT_CREDIT"), c("AMT_ANNUITY")))
        .withColumn("GOODS_MINUS_CREDIT", c("AMT_GOODS_PRICE") - c("AMT_CREDIT"))
        .withColumn("EMPLOYED_TO_AGE", pandas_divide(c("DAYS_EMPLOYED"), c("DAYS_BIRTH")))
        .withColumn("INCOME_PER_PERSON", pandas_divide(c("AMT_INCOME_TOTAL"), c("CNT_FAM_MEMBERS")))
        .withColumn("AGE_AT_LOAN_END", (-c("DAYS_BIRTH")) / 365 + c("TERM"))
    )

    # 4. Tổng hợp 3 điểm tín dụng ngoài THEO TỪNG DÒNG (không phải phép gộp theo nhóm), bỏ qua giá trị thiếu như pandas
    scores = [c(name) for name in EXTERNAL_SCORES]
    known_count = sum(s.isNotNull().cast("int") for s in scores)
    score_sum = sum(F.coalesce(s, F.lit(0.0)) for s in scores)
    frame = frame.withColumn("EXT_SOURCE_MEAN", F.when(known_count > 0, F.try_divide(score_sum, known_count)))
    squared_deviations = sum(F.when(s.isNotNull(), (s - c("EXT_SOURCE_MEAN")) ** 2).otherwise(0.0) for s in scores)
    frame = (
        frame
        # least / greatest của Spark bỏ qua null, chỉ trả null khi cả 3 đều null (giống min/max của pandas)
        .withColumn("EXT_SOURCE_MIN", F.least(*scores))
        .withColumn("EXT_SOURCE_MAX", F.greatest(*scores))
        # Độ lệch chuẩn mẫu (ddof=1) trên các giá trị có mặt; cần ít nhất 2 giá trị, ngược lại null
        .withColumn("EXT_SOURCE_STD", F.when(known_count >= 2, F.sqrt(F.try_divide(squared_deviations, known_count - 1))))
        # Tích chỉ tính khi đủ cả 3 điểm (min_count=3)
        .withColumn("EXT_SOURCE_PROD", F.when(known_count == 3, scores[0] * scores[1] * scores[2]))
    )

    # 7. One-hot. NaN -> tất cả cột dummy = 0 (như get_dummies); chỉ tạo những cột mà mô hình dùng
    dummies = dict(CONTRACT_DUMMIES)
    dummies.update(one_hot_specs(one_hot_names))
    one_hot_columns = [
        F.when(F.regexp_replace(c(source), SANITIZE_PATTERN, "_") == F.lit(value), 1).otherwise(0).cast("tinyint").alias(name)
        for name, (source, value) in dummies.items()
    ]
    return frame.select("*", *one_hot_columns).drop(*ONE_HOT_SOURCES)


def safe_divide(numerator, denominator):
    """main.ipynb cell 40 `safe_div`: mẫu bằng 0 hoặc thiếu -> 0 (khác pandas_divide: ở đây KHÔNG ra inf/NaN)"""
    return F.coalesce(F.try_divide(numerator, denominator), F.lit(0.0))


def bureau_balance_loan_features(bureau_balance):
    """main.ipynb cell 36: gom lịch sử tháng về mỗi khoản vay một dòng. Nhận bảng đã qua clean_bureau_balance."""
    c = F.col
    in_last_12_months = c("MONTHS_BALANCE") >= -11
    in_last_24_months = c("MONTHS_BALANCE") >= -23
    # DPD thiếu (trạng thái C, X) so sánh cho null; pandas coi NaN >= 1 là False nên phải đổi null -> False
    late = F.coalesce(c("DPD") >= 1, F.lit(False))
    seriously_late = F.coalesce(c("DPD") >= 2, F.lit(False))

    grouped = bureau_balance.groupBy("SK_ID_BUREAU").agg(
        F.count(F.lit(1)).alias("BB_N_MONTHS"),
        # max / avg của Spark bỏ qua null như skipna của pandas; cả nhóm null thì ra null
        F.max(F.when(in_last_12_months, c("DPD"))).alias("BB_WORST_12M"),
        F.max(F.when(in_last_24_months, c("DPD"))).alias("BB_WORST_24M"),
        F.max("DPD").alias("BB_WORST_EVER"),
        F.sum(F.when(late & in_last_12_months, 1).otherwise(0)).alias("BB_N_DPD1_12M"),
        F.sum(F.when(seriously_late & in_last_24_months, 1).otherwise(0)).alias("BB_N_DPD2_24M"),
        F.sum(F.when(late, 1).otherwise(0)).alias("BB_N_DPD1_EVER"),
        F.max(F.when(late, c("MONTHS_BALANCE"))).alias("BB_LAST_DPD"),
        F.avg("IS_X").alias("BB_SHARE_X"),
        F.avg("IS_C").alias("BB_SHARE_C"),
    )
    return (
        grouped
        .withColumn("BB_MONTHS_SINCE_DPD", -c("BB_LAST_DPD"))
        .withColumn("BB_EVER_DPD5", F.coalesce(c("BB_WORST_EVER") == 5, F.lit(False)).cast("tinyint"))
        .withColumn("BB_SHARE_DPD1", c("BB_N_DPD1_EVER") / c("BB_N_MONTHS"))
        .select(*BB_LOAN_COLUMNS)
    )


def bureau_customer_features(bureau, bb_loan_features):
    """main.ipynb cell 39 + 40: ghép lịch sử tháng vào bureau, tạo cờ ở cấp khoản vay, gom về mỗi khách hàng một dòng.

    Nhận bảng bureau đã qua clean_bureau và kết quả của bureau_balance_loan_features. Chưa điền giá trị thiếu
    cho khách hàng không có CIC: việc đó làm ở merge_and_fill, giống cell 41.
    """
    c = F.col
    # --- Cấp khoản vay (cell 39) ---
    loans = bureau.join(bb_loan_features, on="SK_ID_BUREAU", how="left")
    # HAS_BB phải tính TRƯỚC khi điền 0 vào BB_N_MONTHS
    loans = loans.withColumn("HAS_BB", c("BB_N_MONTHS").isNotNull().cast("tinyint"))
    loans = loans.withColumn("BB_MONTHS_SINCE_DPD", F.coalesce(c("BB_MONTHS_SINCE_DPD"), F.lit(9999)))
    for name in ["BB_N_MONTHS", "BB_WORST_12M", "BB_WORST_24M", "BB_WORST_EVER", "BB_N_DPD1_12M", "BB_N_DPD2_24M",
                 "BB_SHARE_X", "BB_EVER_DPD5"]:
        loans = loans.withColumn(name, F.coalesce(c(name), F.lit(0)))
    # BB_SHARE_C và BB_SHARE_DPD1 GIỮ null (khoản vay không có lịch sử tháng)

    is_active = c("CREDIT_ACTIVE") == "Active"
    is_currency_1 = c("CREDIT_CURRENCY") == "currency 1"
    is_credit_card = c("CREDIT_TYPE") == "Credit card"
    has_known_end_dates = c("DAYS_ENDDATE_FACT").isNotNull() & c("DAYS_CREDIT_ENDDATE").isNotNull()
    is_valid_active_card = is_active & is_currency_1 & is_credit_card & c("AMT_CREDIT_SUM_LIMIT").isNotNull()

    def flag(condition):
        """cờ 0/1; điều kiện null (so sánh với giá trị thiếu) tính là False như pandas"""
        return F.coalesce(condition, F.lit(False)).cast("tinyint")

    def amount_where(condition, value):
        """pandas `.where(điều kiện).fillna(0)`"""
        return F.coalesce(F.when(condition, value), F.lit(0.0))

    loans = loans.select(
        "*",
        flag(is_active).alias("IS_ACTIVE"),
        flag(c("CREDIT_ACTIVE") == "Closed").alias("IS_CLOSED"),
        flag(c("CREDIT_ACTIVE").isin("Sold", "Bad debt")).alias("IS_BAD"),
        flag(c("DAYS_CREDIT") >= -182).alias("NEW_6M"),
        flag(c("DAYS_CREDIT") >= -365).alias("NEW_12M"),
        flag(is_active & (c("DAYS_CREDIT_ENDDATE") < 0)).alias("ACT_PAST_END"),
        flag(has_known_end_dates).alias("CLOSED_KNOWN"),
        flag(has_known_end_dates & (c("DAYS_ENDDATE_FACT") > c("DAYS_CREDIT_ENDDATE"))).alias("CLOSED_LATE"),
        flag(c("CREDIT_DAY_OVERDUE") > 0).alias("IS_OVERDUE_NOW"),
        flag(c("CNT_CREDIT_PROLONG") > 0).alias("IS_PROLONG"),
        flag(c("CREDIT_TYPE") == "Mortgage").alias("IS_MORTGAGE"),
        flag(c("CREDIT_TYPE") == "Car loan").alias("IS_CAR"),
        flag(c("CREDIT_TYPE") == "Microloan").alias("IS_MICRO"),
        flag(is_credit_card).alias("IS_CARD"),
        # Số tiền có điều kiện: chỉ cộng khoản đang hoạt động, cùng loại tiền tệ chính
        amount_where(is_active & is_currency_1, c("AMT_CREDIT_SUM_DEBT")).alias("DEBT_ACT"),
        amount_where(is_active & is_currency_1, c("AMT_CREDIT_SUM")).alias("CREDIT_ACT"),
        amount_where(is_active & is_currency_1, c("AMT_ANNUITY")).alias("ANN_ACT"),
        amount_where(is_active & is_currency_1, c("AMT_CREDIT_SUM_OVERDUE")).alias("OVD_ACT"),
        amount_where(is_valid_active_card, c("AMT_CREDIT_SUM_DEBT")).alias("CARD_DEBT"),
        amount_where(is_valid_active_card, c("AMT_CREDIT_SUM_LIMIT")).alias("CARD_LIMIT"),
        amount_where(is_currency_1, c("AMT_CREDIT_MAX_OVERDUE")).alias("MAX_OVD_AMT"),
    )

    # --- Gom về cấp khách hàng (cell 40) ---
    grouped = loans.groupBy("SK_ID_CURR").agg(
        F.count(F.lit(1)).alias("BUR_N_LOANS"),
        F.sum("IS_ACTIVE").alias("BUR_N_ACTIVE"),
        F.sum("IS_CLOSED").alias("BUR_N_CLOSED"),
        F.max("IS_BAD").alias("BUR_EVER_BAD"),
        F.sum("NEW_6M").alias("BUR_N_NEW_6M"),
        F.sum("NEW_12M").alias("BUR_N_NEW_12M"),
        F.sum("ACT_PAST_END").alias("BUR_N_ACT_PAST_END"),
        F.sum("CLOSED_KNOWN").alias("BUR_N_CLOSED_KNOWN"),
        F.sum("CLOSED_LATE").alias("BUR_N_CLOSED_LATE"),
        F.sum("IS_OVERDUE_NOW").alias("BUR_N_OVERDUE_NOW"),
        F.max("CREDIT_DAY_OVERDUE").alias("BUR_MAX_DPD_NOW"),
        F.max("IS_PROLONG").alias("BUR_EVER_PROLONG"),
        F.max("IS_MORTGAGE").alias("BUR_HAS_MORTGAGE"),
        F.max("IS_CAR").alias("BUR_HAS_CAR"),
        F.sum("IS_MICRO").alias("BUR_N_MICRO"),
        F.sum("IS_CARD").alias("BUR_N_CARD"),
        F.max("MAX_OVD_AMT").alias("BUR_MAX_OVD_AMT"),
        F.min("DAYS_CREDIT").alias("_OLDEST"),
        F.max("DAYS_CREDIT").alias("_NEWEST"),
        F.max("DAYS_ENDDATE_FACT").alias("_LAST_CLOSED"),
        F.max("BB_WORST_12M").alias("BB_WORST_12M"),
        F.max("BB_WORST_24M").alias("BB_WORST_24M"),
        F.max("BB_WORST_EVER").alias("BB_WORST_EVER"),
        F.max("BB_EVER_DPD5").alias("BB_EVER_DPD5"),
        F.min("BB_MONTHS_SINCE_DPD").alias("BB_MONTHS_SINCE_DPD"),
        F.avg("BB_SHARE_X").alias("BB_SHARE_X"),
        F.sum("BB_N_DPD1_12M").alias("BB_N_DPD1_12M"),
        F.sum("BB_N_DPD2_24M").alias("BB_N_DPD2_24M"),
        F.max("HAS_BB").alias("HAS_BB"),
        F.sum("DEBT_ACT").alias("BUR_ACTIVE_DEBT"),
        F.sum("CREDIT_ACT").alias("BUR_ACTIVE_CREDIT"),
        F.sum("ANN_ACT").alias("BUR_ACTIVE_ANNUITY"),
        F.sum("OVD_ACT").alias("BUR_ACTIVE_OVD_AMT"),
        F.sum("CARD_DEBT").alias("BUR_CARD_DEBT"),
        F.sum("CARD_LIMIT").alias("BUR_CARD_LIMIT"),
    )
    grouped = (
        grouped
        # Số tháng = -số ngày / 30.4; chưa từng đóng khoản vay nào (không có ngày đóng) -> 9999 = rất lâu
        .withColumn("BUR_HIST_MONTHS", F.coalesce(-c("_OLDEST") / 30.4, F.lit(9999.0)))
        .withColumn("BUR_MONTHS_SINCE_NEW", F.coalesce(-c("_NEWEST") / 30.4, F.lit(9999.0)))
        .withColumn("BUR_MONTHS_SINCE_CLOSED", F.coalesce(-c("_LAST_CLOSED") / 30.4, F.lit(9999.0)))
        .withColumn("BUR_ACTIVE_SHARE", safe_divide(c("BUR_N_ACTIVE"), c("BUR_N_LOANS")))
        .withColumn("BUR_CLOSED_LATE_RATE", safe_divide(c("BUR_N_CLOSED_LATE"), c("BUR_N_CLOSED_KNOWN")))
        .withColumn("BUR_DEBT_TO_CREDIT", safe_divide(c("BUR_ACTIVE_DEBT"), c("BUR_ACTIVE_CREDIT")))
        .withColumn("BUR_CARD_UTIL", safe_divide(c("BUR_CARD_DEBT"), c("BUR_CARD_LIMIT")))
        .withColumn("BUR_OVD_TO_DEBT", safe_divide(c("BUR_ACTIVE_OVD_AMT"), c("BUR_ACTIVE_DEBT")))
    )

    # Thống kê mean/max: GIỮ null khi không có dữ liệu. pandas sum của cả nhóm NaN ra 0, Spark ra null, nên sum phải coalesce
    all_loans_stats = loans.groupBy("SK_ID_CURR").agg(
        F.avg("DAYS_CREDIT").alias("BUR_DAYS_CREDIT_MEAN"),
        F.avg("DAYS_CREDIT_ENDDATE").alias("BUR_DAYS_CREDIT_ENDDATE_MEAN"),
        F.max("DAYS_CREDIT_ENDDATE").alias("BUR_DAYS_CREDIT_ENDDATE_MAX"),
        F.avg("DAYS_CREDIT_UPDATE").alias("BUR_DAYS_CREDIT_UPDATE_MEAN"),
        F.avg("DAYS_ENDDATE_FACT").alias("BUR_DAYS_ENDDATE_FACT_MEAN"),
        F.avg("AMT_CREDIT_SUM").alias("BUR_AMT_CREDIT_SUM_MEAN"),
        F.max("AMT_CREDIT_SUM").alias("BUR_AMT_CREDIT_SUM_MAX"),
        F.coalesce(F.sum("AMT_CREDIT_SUM"), F.lit(0.0)).alias("BUR_AMT_CREDIT_SUM_TOTAL"),
        F.avg("AMT_CREDIT_SUM_DEBT").alias("BUR_AMT_DEBT_MEAN"),
        F.max("AMT_CREDIT_SUM_DEBT").alias("BUR_AMT_DEBT_MAX"),
        F.avg("AMT_CREDIT_MAX_OVERDUE").alias("BUR_AMT_MAX_OVERDUE_MEAN"),
        F.coalesce(F.sum("CNT_CREDIT_PROLONG"), F.lit(0)).alias("BUR_CNT_PROLONG_SUM"),
        F.avg("BB_N_MONTHS").alias("BUR_BB_N_MONTHS_MEAN"),
        F.avg("BB_SHARE_C").alias("BUR_BB_SHARE_C_MEAN"),
        F.avg("BB_SHARE_DPD1").alias("BUR_BB_SHARE_DPD1_MEAN"),
    )
    active_stats = loans.filter(c("IS_ACTIVE") == 1).groupBy("SK_ID_CURR").agg(
        F.avg("DAYS_CREDIT").alias("BUR_ACT_DAYS_CREDIT_MEAN"),
        F.avg("DAYS_CREDIT_ENDDATE").alias("BUR_ACT_DAYS_CREDIT_ENDDATE_MEAN"),
        F.avg("DAYS_CREDIT_UPDATE").alias("BUR_ACT_DAYS_UPDATE_MEAN"),
        F.avg("AMT_CREDIT_SUM").alias("BUR_ACT_AMT_CREDIT_SUM_MEAN"),
    )
    closed_stats = loans.filter(c("IS_CLOSED") == 1).groupBy("SK_ID_CURR").agg(
        F.avg("DAYS_CREDIT").alias("BUR_CLS_DAYS_CREDIT_MEAN"),
        F.max("DAYS_CREDIT").alias("BUR_CLS_DAYS_CREDIT_MAX"),
        F.max("DAYS_ENDDATE_FACT").alias("BUR_CLS_DAYS_ENDDATE_FACT_MAX"),
        F.avg("AMT_CREDIT_SUM").alias("BUR_CLS_AMT_CREDIT_SUM_MEAN"),
    )
    return (
        grouped
        .join(all_loans_stats, on="SK_ID_CURR", how="left")
        .join(active_stats, on="SK_ID_CURR", how="left")
        .join(closed_stats, on="SK_ID_CURR", how="left")
        .select(*BUREAU_CUSTOMER_COLUMNS)
    )


def merge_and_fill(application, bureau_customer, feature_names):
    """main.ipynb cell 41: ghép bureau vào application, điền giá trị thiếu, tạo biến tương tác, lọc Cash loans,
    rồi chọn đúng 221 đặc trưng theo thứ tự trong metadata.json của mô hình.

    Kết quả giữ thêm SK_ID_CURR và TARGET (nếu có) ở đầu, không nằm trong 221 đặc trưng.
    """
    c = F.col
    frame = application.join(bureau_customer, on="SK_ID_CURR", how="left")
    # HAS_BUREAU phải tính TRƯỚC khi điền 0 vào BUR_N_LOANS
    frame = frame.withColumn("HAS_BUREAU", c("BUR_N_LOANS").isNotNull().cast("tinyint"))
    frame = frame.withColumn("HAS_BB", F.coalesce(c("HAS_BB"), F.lit(0)))

    # Khách hàng không có CIC. Điền THEO TÊN CỘT như notebook:
    #   chứa "MONTHS_SINCE" -> 9999 (chưa từng xảy ra = rất lâu); cột còn lại -> 0;
    #   riêng 23 cột thống kê mean/max giữ null (0 sẽ mang nghĩa sai, ví dụ "khoản vay mở ngay hôm nay").
    # Vì quy tắc theo tên, BUR_HIST_MONTHS (không chứa MONTHS_SINCE) được điền 0 chứ không phải 9999.
    filled = []
    for name in frame.columns:
        column = F.col(f"`{name}`")
        if name.startswith(("BUR_", "BB_")) and name not in NUMERIC_STATS_COLUMNS:
            column = F.coalesce(column, F.lit(9999 if "MONTHS_SINCE" in name else 0)).alias(name)
        filled.append(column)
    frame = frame.select(*filled)

    # Biến tương tác Application x Bureau
    income = c("AMT_INCOME_TOTAL")
    frame = (
        frame
        .withColumn("BUR_DEBT_TO_INCOME", safe_divide(c("BUR_ACTIVE_DEBT"), income))
        .withColumn("TOTAL_PTI", c("PTI") + safe_divide(c("BUR_ACTIVE_ANNUITY"), income))
        .withColumn("NEW_LOAN_SHARE", safe_divide(c("AMT_CREDIT"), c("AMT_CREDIT") + c("BUR_ACTIVE_DEBT")))
    )

    # Mô hình chỉ học trên Cash loans
    cash = frame.filter(c("`NAME_CONTRACT_TYPE_Cash loans`") == 1)
    identifiers = [name for name in ("SK_ID_CURR", "TARGET") if name in cash.columns]
    return cash.select(*identifiers, *[F.col(f"`{name}`") for name in feature_names])
