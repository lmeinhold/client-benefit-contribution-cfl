FROM python:3.11-bookworm

WORKDIR /usr/src/cfl

COPY requirements* ./
RUN pip install --no-cache-dir -r requirements.txt

COPY experiments .
COPY federated_learning .
COPY __init__.py .
COPY train.py .
COPY models .
COPY scripts .
COPY utils .