FROM python:3.11-slim

WORKDIR /usr/src/cfl

COPY requirements* ./
RUN pip install --no-cache-dir -r requirements.txt

COPY federated_learning federated_learning
COPY datasets datasets
COPY __init__.py __init__.py
COPY *.py .
COPY models models
COPY utils utils
