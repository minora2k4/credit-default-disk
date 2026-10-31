"""Xuất các bảng trung gian của main.ipynb ra Parquet để làm "đáp án chuẩn" khi kiểm tra code Spark.

Chạy các code cell 0-41 của notebook trong bộ nhớ (backend matplotlib không hiển thị), KHÔNG sửa file notebook.
Cell 41 bị thay đúng một dòng ở bản chạy trong bộ nhớ: giữ lại SK_ID_CURR (notebook bỏ cột này trước khi chia
Cash/Revolving) để khớp từng hồ sơ với kết quả Spark.

Cần khoảng 6 GB RAM (bureau_balance 27 triệu dòng) và vài phút. Nên tắt stack Docker trước khi chạy.

Đầu ra trong --out-dir:
  application_features.parquet     bảng application sau cell 9 + 19 (đặc trưng + one-hot), TRƯỚC khi ghép bureau
  bureau_balance_features.parquet  mỗi khoản vay một dòng (cell 36)
  bureau_features.parquet          mỗi khách hàng một dòng, trước bước điền giá trị thiếu (cell 40)
  application_train_cash.parquet   bảng cuối cùng của notebook: SK_ID_CURR + TARGET + 221 đặc trưng
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("MPLBACKEND", "Agg")

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAST_CELL_BEFORE_SPLIT = 40   # cell 41 ghép bureau vào application, tách Cash/Revolving rồi `del` các bảng trung gian
SPLIT_CELL = 41
DROP_SPLIT_LINE = 'drop_split_cols = ["NAME_CONTRACT_TYPE_Cash loans", "NAME_CONTRACT_TYPE_Revolving loans", "SK_ID_CURR"]'
KEEP_ID_LINE = 'drop_split_cols = ["NAME_CONTRACT_TYPE_Cash loans", "NAME_CONTRACT_TYPE_Revolving loans"]'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(REPOSITORY_ROOT, "de_pipeline", "work", "reference"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(REPOSITORY_ROOT)   # notebook đọc "data/*.csv" theo đường dẫn tương đối

    with open("main.ipynb", encoding="utf-8") as notebook_file:
        cells = json.load(notebook_file)["cells"]

    namespace = {"__name__": "__notebook__"}
    from IPython.display import display   # notebook gọi display() ở vài cell
    namespace["display"] = display

    def run_cell(index, source=None):
        code = source if source is not None else "".join(cells[index]["source"])
        started = time.time()
        exec(compile(code, f"main.ipynb[cell {index}]", "exec"), namespace)
        print(f"  cell {index:>2} xong ({time.time() - started:.1f}s)", flush=True)

    print("1. Chạy cell 0-40 của notebook")
    for index in range(LAST_CELL_BEFORE_SPLIT + 1):
        if cells[index]["cell_type"] == "code":
            run_cell(index)

    print("2. Xuất 3 bảng trung gian")
    namespace["application_train"].to_parquet(os.path.join(args.out_dir, "application_features.parquet"), index=False)
    namespace["bureau_balance_features"].to_parquet(os.path.join(args.out_dir, "bureau_balance_features.parquet"), index=False)
    namespace["bureau_features"].to_parquet(os.path.join(args.out_dir, "bureau_features.parquet"), index=False)
    print(f"   application {namespace['application_train'].shape} | bureau_balance_features {namespace['bureau_balance_features'].shape} "
          f"| bureau_features {namespace['bureau_features'].shape}")

    print("3. Chạy cell 41 (ghép bureau, tách Cash/Revolving) với SK_ID_CURR được giữ lại")
    split_source = "".join(cells[SPLIT_CELL]["source"])
    assert DROP_SPLIT_LINE in split_source, "Cell 41 đã đổi, cập nhật DROP_SPLIT_LINE"
    run_cell(SPLIT_CELL, split_source.replace(DROP_SPLIT_LINE, KEEP_ID_LINE))

    cash = namespace["application_train_cash"]
    ids = cash.pop("SK_ID_CURR")
    cash.insert(0, "SK_ID_CURR", ids)
    cash.to_parquet(os.path.join(args.out_dir, "application_train_cash.parquet"), index=False)
    feature_count = cash.shape[1] - 2   # trừ SK_ID_CURR và TARGET
    print(f"4. application_train_cash: {cash.shape[0]} dòng, {feature_count} đặc trưng (+ SK_ID_CURR, TARGET)")
    assert cash.shape[0] == 278220 and feature_count == 221, "Số dòng/đặc trưng không đúng như notebook"
    print(f"Đã ghi 4 file vào {args.out_dir}")


if __name__ == "__main__":
    main()
