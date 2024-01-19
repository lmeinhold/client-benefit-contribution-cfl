FROM python:3.11-bookworm

ENV MCFL_EXPERIMENTS=all

WORKDIR /usr/src/mcfl

COPY experiments .
COPY federated_learning .
COPY __init__.py .
COPY run.py .
COPY models .
COPY requirements* ./
COPY scripts .
COPY utils .


RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./scripts/run.py", "-e", "$MCFL_EXPERIMENTS" ]