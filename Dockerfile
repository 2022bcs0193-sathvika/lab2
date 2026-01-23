FROM python:3.11-slim

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY mlops_lab1.py .
COPY dataset/ dataset/
COPY output/ output/

CMD ["python", "mlops_lab1.py"]
