"""Tái tạo chính xác tập test hold-out của main.ipynb (mục 4-5) mà không cần chạy notebook.

Phép chia train_test_split chỉ phụ thuộc số dòng và nhãn, nên nếu lọc dòng đúng thứ tự như notebook
thì các dòng rơi vào tập test là hoàn toàn giống notebook.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Notebook cell 9: loại hồ sơ thiếu một trong 4 cột số tiền này
REQUIRED_AMOUNT_COLUMNS = ["AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]


def build_holdout_ids(application_csv_path):
    """Trả về DataFrame [SK_ID_CURR, TARGET, split] cho toàn bộ hồ sơ Cash loans.

    Các bước giữ đúng thứ tự như notebook: dropna -> lọc Cash loans -> reset_index -> chia 80/20 stratify.
    """
    application = pd.read_csv(application_csv_path, usecols=["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE"] + REQUIRED_AMOUNT_COLUMNS)
    application = application.dropna(subset=REQUIRED_AMOUNT_COLUMNS)
    cash = application[application["NAME_CONTRACT_TYPE"] == "Cash loans"].reset_index(drop=True)

    train_index, test_index = train_test_split(
        cash.index, test_size=0.2, random_state=42, stratify=cash["TARGET"]
    )
    holdout_ids = cash[["SK_ID_CURR", "TARGET"]].copy()
    holdout_ids["split"] = "train"
    holdout_ids.loc[test_index, "split"] = "test"
    return holdout_ids
