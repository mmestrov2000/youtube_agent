FROM python:3.11-slim

WORKDIR /app

COPY req.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r req.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]