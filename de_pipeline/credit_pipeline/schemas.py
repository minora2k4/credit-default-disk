"""Schema (tên cột + kiểu) của 3 bảng CSV gốc, sinh từ chính các file CSV.

Spark đọc CSV theo VỊ TRÍ cột khi đã khai báo schema, nên phải liệt kê đủ và đúng thứ tự mọi cột trong file.
Khai báo sẵn schema nhanh hơn inferSchema (không phải quét file 2 lần) và kiểu dữ liệu khớp với pandas.
"""
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

_TYPES = {"long": LongType(), "double": DoubleType(), "string": StringType()}


def _struct(columns):
    return StructType([StructField(name, _TYPES[kind], True) for name, kind in columns])

# 122 cột
APPLICATION_COLUMNS = [
    ("SK_ID_CURR", "long"), ("TARGET", "long"), ("NAME_CONTRACT_TYPE", "string"), ("CODE_GENDER", "string"),
    ("FLAG_OWN_CAR", "string"), ("FLAG_OWN_REALTY", "string"), ("CNT_CHILDREN", "long"),
    ("AMT_INCOME_TOTAL", "double"), ("AMT_CREDIT", "double"), ("AMT_ANNUITY", "double"),
    ("AMT_GOODS_PRICE", "double"), ("NAME_TYPE_SUITE", "string"), ("NAME_INCOME_TYPE", "string"),
    ("NAME_EDUCATION_TYPE", "string"), ("NAME_FAMILY_STATUS", "string"), ("NAME_HOUSING_TYPE", "string"),
    ("REGION_POPULATION_RELATIVE", "double"), ("DAYS_BIRTH", "long"), ("DAYS_EMPLOYED", "long"),
    ("DAYS_REGISTRATION", "double"), ("DAYS_ID_PUBLISH", "long"), ("OWN_CAR_AGE", "double"), ("FLAG_MOBIL", "long"),
    ("FLAG_EMP_PHONE", "long"), ("FLAG_WORK_PHONE", "long"), ("FLAG_CONT_MOBILE", "long"), ("FLAG_PHONE", "long"),
    ("FLAG_EMAIL", "long"), ("OCCUPATION_TYPE", "string"), ("CNT_FAM_MEMBERS", "double"),
    ("REGION_RATING_CLIENT", "long"), ("REGION_RATING_CLIENT_W_CITY", "long"),
    ("WEEKDAY_APPR_PROCESS_START", "string"), ("HOUR_APPR_PROCESS_START", "long"),
    ("REG_REGION_NOT_LIVE_REGION", "long"), ("REG_REGION_NOT_WORK_REGION", "long"),
    ("LIVE_REGION_NOT_WORK_REGION", "long"), ("REG_CITY_NOT_LIVE_CITY", "long"), ("REG_CITY_NOT_WORK_CITY", "long"),
    ("LIVE_CITY_NOT_WORK_CITY", "long"), ("ORGANIZATION_TYPE", "string"), ("EXT_SOURCE_1", "double"),
    ("EXT_SOURCE_2", "double"), ("EXT_SOURCE_3", "double"), ("APARTMENTS_AVG", "double"),
    ("BASEMENTAREA_AVG", "double"), ("YEARS_BEGINEXPLUATATION_AVG", "double"), ("YEARS_BUILD_AVG", "double"),
    ("COMMONAREA_AVG", "double"), ("ELEVATORS_AVG", "double"), ("ENTRANCES_AVG", "double"),
    ("FLOORSMAX_AVG", "double"), ("FLOORSMIN_AVG", "double"), ("LANDAREA_AVG", "double"),
    ("LIVINGAPARTMENTS_AVG", "double"), ("LIVINGAREA_AVG", "double"), ("NONLIVINGAPARTMENTS_AVG", "double"),
    ("NONLIVINGAREA_AVG", "double"), ("APARTMENTS_MODE", "double"), ("BASEMENTAREA_MODE", "double"),
    ("YEARS_BEGINEXPLUATATION_MODE", "double"), ("YEARS_BUILD_MODE", "double"), ("COMMONAREA_MODE", "double"),
    ("ELEVATORS_MODE", "double"), ("ENTRANCES_MODE", "double"), ("FLOORSMAX_MODE", "double"),
    ("FLOORSMIN_MODE", "double"), ("LANDAREA_MODE", "double"), ("LIVINGAPARTMENTS_MODE", "double"),
    ("LIVINGAREA_MODE", "double"), ("NONLIVINGAPARTMENTS_MODE", "double"), ("NONLIVINGAREA_MODE", "double"),
    ("APARTMENTS_MEDI", "double"), ("BASEMENTAREA_MEDI", "double"), ("YEARS_BEGINEXPLUATATION_MEDI", "double"),
    ("YEARS_BUILD_MEDI", "double"), ("COMMONAREA_MEDI", "double"), ("ELEVATORS_MEDI", "double"),
    ("ENTRANCES_MEDI", "double"), ("FLOORSMAX_MEDI", "double"), ("FLOORSMIN_MEDI", "double"),
    ("LANDAREA_MEDI", "double"), ("LIVINGAPARTMENTS_MEDI", "double"), ("LIVINGAREA_MEDI", "double"),
    ("NONLIVINGAPARTMENTS_MEDI", "double"), ("NONLIVINGAREA_MEDI", "double"), ("FONDKAPREMONT_MODE", "string"),
    ("HOUSETYPE_MODE", "string"), ("TOTALAREA_MODE", "double"), ("WALLSMATERIAL_MODE", "string"),
    ("EMERGENCYSTATE_MODE", "string"), ("OBS_30_CNT_SOCIAL_CIRCLE", "double"), ("DEF_30_CNT_SOCIAL_CIRCLE", "double"),
    ("OBS_60_CNT_SOCIAL_CIRCLE", "double"), ("DEF_60_CNT_SOCIAL_CIRCLE", "double"),
    ("DAYS_LAST_PHONE_CHANGE", "double"), ("FLAG_DOCUMENT_2", "long"), ("FLAG_DOCUMENT_3", "long"),
    ("FLAG_DOCUMENT_4", "long"), ("FLAG_DOCUMENT_5", "long"), ("FLAG_DOCUMENT_6", "long"),
    ("FLAG_DOCUMENT_7", "long"), ("FLAG_DOCUMENT_8", "long"), ("FLAG_DOCUMENT_9", "long"),
    ("FLAG_DOCUMENT_10", "long"), ("FLAG_DOCUMENT_11", "long"), ("FLAG_DOCUMENT_12", "long"),
    ("FLAG_DOCUMENT_13", "long"), ("FLAG_DOCUMENT_14", "long"), ("FLAG_DOCUMENT_15", "long"),
    ("FLAG_DOCUMENT_16", "long"), ("FLAG_DOCUMENT_17", "long"), ("FLAG_DOCUMENT_18", "long"),
    ("FLAG_DOCUMENT_19", "long"), ("FLAG_DOCUMENT_20", "long"), ("FLAG_DOCUMENT_21", "long"),
    ("AMT_REQ_CREDIT_BUREAU_HOUR", "double"), ("AMT_REQ_CREDIT_BUREAU_DAY", "double"),
    ("AMT_REQ_CREDIT_BUREAU_WEEK", "double"), ("AMT_REQ_CREDIT_BUREAU_MON", "double"),
    ("AMT_REQ_CREDIT_BUREAU_QRT", "double"), ("AMT_REQ_CREDIT_BUREAU_YEAR", "double"),
]
APPLICATION_SCHEMA = _struct(APPLICATION_COLUMNS)

# 17 cột
BUREAU_COLUMNS = [
    ("SK_ID_CURR", "long"), ("SK_ID_BUREAU", "long"), ("CREDIT_ACTIVE", "string"), ("CREDIT_CURRENCY", "string"),
    ("DAYS_CREDIT", "long"), ("CREDIT_DAY_OVERDUE", "long"), ("DAYS_CREDIT_ENDDATE", "double"),
    ("DAYS_ENDDATE_FACT", "double"), ("AMT_CREDIT_MAX_OVERDUE", "double"), ("CNT_CREDIT_PROLONG", "long"),
    ("AMT_CREDIT_SUM", "double"), ("AMT_CREDIT_SUM_DEBT", "double"), ("AMT_CREDIT_SUM_LIMIT", "double"),
    ("AMT_CREDIT_SUM_OVERDUE", "double"), ("CREDIT_TYPE", "string"), ("DAYS_CREDIT_UPDATE", "long"),
    ("AMT_ANNUITY", "double"),
]
BUREAU_SCHEMA = _struct(BUREAU_COLUMNS)

# 3 cột
BUREAU_BALANCE_COLUMNS = [
    ("SK_ID_BUREAU", "long"), ("MONTHS_BALANCE", "long"), ("STATUS", "string"),
]
BUREAU_BALANCE_SCHEMA = _struct(BUREAU_BALANCE_COLUMNS)
