FROM python:3.11-alpine

WORKDIR /usr/src/mcfl

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./scripts/run_experiments.py" ]