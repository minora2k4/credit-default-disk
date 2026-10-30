.DEFAULT_GOAL := help
.PHONY: help hdfs-ingest hdfs-verify topics sim-pool-real sim-pool-synth simulate

NN := docker compose exec -T namenode
TABLES := application_train bureau bureau_balance
MODEL_DIR := /models/credit_default/lightgbm/v1

# Tham số của `make simulate` (đổi bằng cách gõ ví dụ: make simulate SOURCE=synthetic RATE=300 LIMIT=0)
SOURCE ?= real-test
RATE ?= 100
LIMIT ?= 1000
OUTCOME_DELAY ?= 15

help: ## Liệt kê các target
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

hdfs-ingest: ## Đưa CSV gốc + mô hình ONNX lên HDFS (chạy lại bao nhiêu lần cũng an toàn)
	$(NN) hdfs dfs -mkdir -p /datalake/silver /datalake/gold /datalake/reference /checkpoints $(MODEL_DIR)
	@for t in $(TABLES); do \
		$(NN) hdfs dfs -mkdir -p /datalake/bronze/batch/$$t && \
		$(NN) hdfs dfs -put -f /raw/$$t.csv /datalake/bronze/batch/$$t/ && echo "put $$t.csv" || exit 1; \
	done
	$(NN) hdfs dfs -put -f /artifacts/LightGBM/model.onnx /artifacts/LightGBM/metadata.json $(MODEL_DIR)/
	$(NN) hdfs dfs -ls -R -h /datalake/bronze $(MODEL_DIR)

hdfs-verify: ## So md5 dữ liệu trên HDFS với file gốc + kiểm tra fsck
	$(NN) hdfs dfsadmin -safemode wait
	@for t in $(TABLES); do \
		a=$$($(NN) bash -c "hdfs dfs -cat /datalake/bronze/batch/$$t/$$t.csv | md5sum" | cut -d' ' -f1); \
		b=$$(md5sum ../data/$$t.csv | cut -d' ' -f1); \
		[ "$$a" = "$$b" ] && echo "$$t: md5 OK" || { echo "$$t: md5 LỆCH"; exit 1; }; \
	done
	@for m in model.onnx metadata.json; do \
		a=$$($(NN) bash -c "hdfs dfs -cat $(MODEL_DIR)/$$m | md5sum" | cut -d' ' -f1); \
		b=$$(md5sum ../mlops_artifacts/LightGBM/$$m | cut -d' ' -f1); \
		[ "$$a" = "$$b" ] && echo "$$m: md5 OK" || { echo "$$m: md5 LỆCH"; exit 1; }; \
	done
	$(NN) hdfs fsck /datalake | grep -E 'Total blocks|Status'
	$(NN) hdfs fsck /models | grep -E 'Total blocks|Status'

topics: ## Tạo topic loan_applications, loan_outcomes (chạy lại an toàn)
	docker compose run --rm kafka-init

sim-pool-real: ## Tạo pool phát lại 55,644 hồ sơ THẬT của tập test + holdout_ids (work/)
	docker compose --profile sim run --rm simulator python simulator/build_pool.py

sim-pool-synth: ## Sinh pool dữ liệu TỔNG HỢP bằng SDV theo phân phối toàn bộ Cash loans (~5 phút)
	docker compose --profile sim run --rm synthesizer python simulator/synthesize_pool.py

simulate: ## Bắn hồ sơ vào Kafka. Tham số: SOURCE=real-test|synthetic RATE=hồ sơ/giây LIMIT=số hồ sơ (0=hết) OUTCOME_DELAY=giây
	docker compose --profile sim run --rm simulator python simulator/producer.py --source $(SOURCE) --rate $(RATE) --limit $(LIMIT) --outcome-delay $(OUTCOME_DELAY)
