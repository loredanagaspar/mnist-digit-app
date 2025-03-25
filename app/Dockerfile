FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    streamlit \
    torch \
    torchvision \
    pandas \
    psycopg2-binary \
    streamlit-drawable-canvas

CMD ["streamlit", "run", "app.py", "--server.enableCORS=false"]
