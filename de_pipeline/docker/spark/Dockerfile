# Một image cho master, worker và driver: driver và executor phải cùng phiên bản Python và thư viện
# (pandas UDF / mapInPandas gửi dữ liệu giữa JVM và Python worker bằng Arrow).
FROM apache/spark:4.1.3-python3

USER root
COPY requirements.txt /tmp/requirements.txt
# pip 22 của Ubuntu 22.04 đã cũ, nâng lên rồi mới cài
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
 # Image có sẵn pyspark ở /opt/spark/python và py4j dạng .zip; đặt tên cố định để khai báo PYTHONPATH (không dùng glob được trong ENV)
 && ln -s /opt/spark/python/lib/py4j-*-src.zip /opt/spark/python/lib/py4j-src.zip

ENV PATH="${PATH}:/opt/spark/bin:/opt/spark/sbin" \
    PYTHONPATH="/opt/spark/python:/opt/spark/python/lib/py4j-src.zip:/opt/pipeline" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf

USER spark
WORKDIR /opt/pipeline
