#!/usr/bin/env bash
# Tạo các topic của pipeline. Chạy lại bao nhiêu lần cũng an toàn (--if-not-exists).
set -euo pipefail

BOOTSTRAP=kafka:9092
KT=/opt/kafka/bin/kafka-topics.sh

# loan_applications: mỗi message là 1 hồ sơ vay (JSON lồng nhau, có thể tới vài trăm KB)
#   - 3 partition: Spark đọc song song theo partition; key = application_id nên mỗi hồ sơ luôn vào cùng 1 partition
#   - retention 7 ngày (604800000 ms)
#   - max.message.bytes 5 MB (mặc định chỉ 1 MB); producer cũng phải nâng message.max.bytes tương ứng
# loan_outcomes: nhãn vỡ nợ (TARGET) gửi sau một khoảng trễ, message rất nhỏ
for topic in loan_applications loan_outcomes; do
  extra=()
  if [ "$topic" = "loan_applications" ]; then
    extra=(--config max.message.bytes=5242880)
  fi
  "$KT" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$topic" --partitions 3 --replication-factor 1 \
    --config retention.ms=604800000 "${extra[@]}"
done

"$KT" --bootstrap-server "$BOOTSTRAP" --list
