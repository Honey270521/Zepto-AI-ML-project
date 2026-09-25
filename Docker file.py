FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
WORKDIR /app/support_assistant
EXPOSE 7860
CMD ["python", "main.py"]