FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from current dir to /app
COPY . .

EXPOSE 7860

# Fixed: point to main:app since main.py is now in the root of the space
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
# Rebuild trigger: 2026-05-25 19:28
