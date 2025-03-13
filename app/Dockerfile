FROM python:3.12-slim

RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr libmagic-dev  libgl1 libglib2.0-0  tesseract-ocr-all

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./

RUN pip install -r requirements.txt

ENV OCR_AGENT="tesseract"

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "server.address=0.0.0.0"]
