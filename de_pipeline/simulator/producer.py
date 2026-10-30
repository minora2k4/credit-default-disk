"""Bắn hồ sơ vay từ sim pool vào Kafka, kèm nhãn vỡ nợ gửi trễ vào topic riêng.

  loan_applications  key = application_id, value = JSON hồ sơ + event_time (UTC lúc gửi)
  loan_outcomes      key = application_id, value = {application_id, TARGET, outcome_time}, gửi sau --outcome-delay giây
                     (mô phỏng việc chỉ biết hồ sơ có vỡ nợ hay không sau một thời gian)

Chạy trong container simulator (xem `make simulate`). Hai nguồn dữ liệu:
  --source real-test   55,644 hồ sơ thật của tập test (dùng cho gate khớp notebook)
  --source synthetic   dữ liệu sinh bằng SDV
"""
import argparse
import collections
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

APPLICATIONS_TOPIC = "loan_applications"
OUTCOMES_TOPIC = "loan_outcomes"
REPORT_EVERY_SECONDS = 5


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def with_event_time(line, event_time):
    """Chèn event_time vào cuối JSON object mà không phải parse lại cả message vài KB: '...}' -> '...,"event_time":"..."}'"""
    line = line.rstrip()
    assert line.endswith("}")
    return f'{line[:-1]},"event_time":"{event_time}"}}'


def build_producer(bootstrap_servers):
    from confluent_kafka import Producer  # import muộn để --dry-run chạy được ở nơi chưa cài confluent-kafka

    return Producer({
        "bootstrap.servers": bootstrap_servers,
        "acks": "all",                       # chờ broker xác nhận ghi xong
        "enable.idempotence": True,          # retry không sinh bản ghi trùng, giữ đúng thứ tự trong partition
        "compression.type": "zstd",
        "linger.ms": 20,                     # gom các message gần nhau thành một batch
        "message.max.bytes": 5 * 1024 * 1024,  # khớp max.message.bytes của topic loan_applications (mặc định chỉ 1 MB)
    })


def check_topics(producer):
    """Fail sớm với thông báo rõ ràng nếu Kafka chưa chạy hoặc chưa tạo topic"""
    try:
        metadata = producer.list_topics(timeout=10)
    except Exception as error:
        sys.exit(f"Không kết nối được Kafka ({error}). Kafka đã chạy chưa? (docker compose up -d kafka)")
    missing = [t for t in (APPLICATIONS_TOPIC, OUTCOMES_TOPIC) if t not in metadata.topics]
    if missing:
        sys.exit(f"Thiếu topic {missing}. Chạy `make topics` trước.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["real-test", "synthetic"], default="real-test")
    parser.add_argument("--pool-dir", default="work/sim_pool")
    parser.add_argument("--rate", type=float, default=100, help="số hồ sơ mỗi giây")
    parser.add_argument("--limit", type=int, default=1000, help="số hồ sơ cần gửi; 0 = gửi hết pool")
    parser.add_argument("--outcome-delay", type=float, default=15, help="số giây trễ trước khi gửi nhãn")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"))
    parser.add_argument("--dry-run", action="store_true", help="không kết nối Kafka, chỉ kiểm tra dữ liệu và tốc độ")
    args = parser.parse_args()

    source_dir = os.path.join(args.pool_dir, args.source.replace("-", "_"))
    outcomes = pd.read_parquet(os.path.join(source_dir, "outcomes.parquet"))
    target_by_id = dict(zip(outcomes["application_id"], outcomes["TARGET"].astype(int)))

    producer = None if args.dry_run else build_producer(args.bootstrap)
    if producer is not None:
        check_topics(producer)

    counters = {"delivered": 0, "errors": 0}

    def on_delivery(error, message):
        if error is not None:
            counters["errors"] += 1
            print(f"LỖI gửi {message.key()}: {error}", file=sys.stderr)
        else:
            counters["delivered"] += 1

    def send(topic, key, value):
        if producer is None:
            return
        while True:
            try:
                producer.produce(topic, key=key, value=value.encode("utf-8"), on_delivery=on_delivery)
                return
            except BufferError:      # hàng đợi nội bộ đầy: chờ gửi bớt rồi thử lại
                producer.poll(0.5)

    pending_outcomes = collections.deque()   # (thời điểm đến hạn, application_id); đến hạn theo đúng thứ tự gửi
    sent_applications = sent_outcomes = 0
    largest_message = 0
    started = last_report = time.monotonic()
    print(f"Nguồn {args.source} | tốc độ {args.rate}/s | giới hạn {args.limit or 'hết pool'} | trễ nhãn {args.outcome_delay}s"
          f"{' | DRY-RUN' if args.dry_run else ' | Kafka ' + args.bootstrap}")

    def flush_due_outcomes(now, force=False):
        nonlocal sent_outcomes
        while pending_outcomes and (force or pending_outcomes[0][0] <= now):
            due, application_id = pending_outcomes.popleft()
            if force and due > time.monotonic():
                time.sleep(due - time.monotonic())
            value = json.dumps({"application_id": application_id, "TARGET": target_by_id[application_id], "outcome_time": utc_now_iso()})
            send(OUTCOMES_TOPIC, application_id, value)
            sent_outcomes += 1

    with gzip.open(os.path.join(source_dir, "applications.jsonl.gz"), "rt", encoding="utf-8") as pool_file:
        for line in pool_file:
            if args.limit and sent_applications >= args.limit:
                break
            # Giữ nhịp: hồ sơ thứ i được gửi vào thời điểm started + i / rate
            wait = started + sent_applications / args.rate - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            application_id = line[line.index('"application_id":"') + 18:].split('"', 1)[0]
            message = with_event_time(line, utc_now_iso())
            largest_message = max(largest_message, len(message))
            send(APPLICATIONS_TOPIC, application_id, message)
            sent_applications += 1
            pending_outcomes.append((time.monotonic() + args.outcome_delay, application_id))
            if producer is not None:
                producer.poll(0)
            now = time.monotonic()
            flush_due_outcomes(now)
            if now - last_report >= REPORT_EVERY_SECONDS:
                print(f"  đã gửi {sent_applications} hồ sơ, {sent_outcomes} nhãn | {sent_applications / (now - started):.0f} hồ sơ/s | lỗi {counters['errors']}")
                last_report = now

    send_elapsed = time.monotonic() - started
    flush_due_outcomes(time.monotonic(), force=True)   # chờ nốt các nhãn chưa đến hạn rồi gửi
    if producer is not None:
        remaining = producer.flush(60)
        if remaining:
            sys.exit(f"Còn {remaining} message chưa gửi được sau 60 giây")
    elapsed = time.monotonic() - started
    print(f"XONG: {sent_applications} hồ sơ + {sent_outcomes} nhãn trong {elapsed:.1f}s "
          f"(gửi hồ sơ {send_elapsed:.1f}s = {sent_applications / max(send_elapsed, 1e-9):.0f} hồ sơ/s) | message lớn nhất {largest_message / 1024:.1f} KB | "
          f"xác nhận {counters['delivered']} | lỗi {counters['errors']}")
    if counters["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
