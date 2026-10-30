FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/pipeline:/opt/pipeline/simulator
COPY requirements.txt /tmp/requirements.txt
# torch bản CPU (~200 MB) thay cho bản mặc định kèm CUDA (vài GB); SDV cần torch dù GaussianCopula không dùng GPU.
#  - Dùng --extra-index-url (thêm index PyTorch cạnh PyPI), KHÔNG dùng --index-url: index PyTorch có metadata sai tên
#    cho jinja2, typing-extensions..., nên nếu thay hẳn PyPI thì pip không cài được các thư viện phụ của torch.
#  - "+cpu" là phiên bản cục bộ chỉ có ở index PyTorch, nên pip không thể nhầm sang bản CUDA trên PyPI.
RUN pip install --no-cache-dir "torch==2.14.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /opt/pipeline
