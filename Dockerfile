FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY output/ ./output
COPY mlops_lab1.py .

CMD ["python", "mlops_lab1.py"]

