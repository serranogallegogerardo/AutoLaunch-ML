FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV API_URL=https://gerakp-pulseml-api.hf.space

# Copy everything from current dir to /app
COPY . .

EXPOSE 7860

# Fixed: point to app.py since it's now in the root of the space
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
