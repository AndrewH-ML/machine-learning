FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

CMD ["python", "demo.py"]
