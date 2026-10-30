# Wrapper ngắn gọn cho lệnh chạy trong container. Dùng: source scripts/aliases.sh
# Sẽ được thêm dần từ giai đoạn 2 (kafka-topics), 3 (spark-submit), 4 (psql).

DE_PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# hdfs dfs -ls /   thay cho   docker compose exec namenode hdfs dfs -ls /
hdfs() { docker compose --project-directory "$DE_PIPELINE_DIR" exec namenode hdfs "$@"; }

# Công cụ dòng lệnh của Kafka, tự gắn --bootstrap-server kafka:9092
#   kafka-topics --list      kafka-consumer-groups --list      kafka-get-offsets --topic loan_applications
_kafka_tool() { local tool="$1"; shift; docker compose --project-directory "$DE_PIPELINE_DIR" exec kafka "/opt/kafka/bin/$tool.sh" --bootstrap-server kafka:9092 "$@"; }
kafka-topics()          { _kafka_tool kafka-topics "$@"; }
kafka-configs()         { _kafka_tool kafka-configs "$@"; }
kafka-get-offsets()     { _kafka_tool kafka-get-offsets "$@"; }
kafka-consumer-groups() { _kafka_tool kafka-consumer-groups "$@"; }
