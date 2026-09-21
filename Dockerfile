# Dockerfile
# ==========
# Builds a lean image that serves app/app.py -- for deploying to Render's
# free tier (or any other Docker-based host: Railway, Fly.io, etc.).
#
# Uses requirements-app.txt (not requirements.txt) so PyTorch and other
# training-only dependencies never end up in the image -- see that file
# for why that's safe. Only models/export/ (the already-exported ONNX
# model + tokenizer) is copied in; models/checkpoints/ and data/ are not
# needed at runtime and are skipped via .dockerignore.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY config.yaml .
COPY src/ src/
COPY app/ app/
COPY models/export/ models/export/

# Render (and most similar hosts) inject $PORT and expect the app to
# listen on 0.0.0.0; app/app.py reads both at startup (see its __main__
# block), with 127.0.0.1:7860 as the local-run default.
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "app/app.py"]
