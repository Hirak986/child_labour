FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "server:app", "--bind", "0.0.0.0:10000"]
