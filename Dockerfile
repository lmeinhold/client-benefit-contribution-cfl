FROM python:3.11-bookworm

WORKDIR /usr/src/mcfl

COPY experiments .
COPY federated_learning .
COPY __init__.py .
COPY train.py .
COPY models .
COPY requirements* ./
COPY scripts .
COPY utils .


RUN pip install --no-cache-dir -r requirements.txt