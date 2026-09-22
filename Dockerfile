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

# Railway (and most similar hosts: Render, Fly.io) inject $PORT and expect
# the app to listen on 0.0.0.0; app/app.py reads both at startup (see its
# __main__ block), with 127.0.0.1:7860 as the local-run default.
#
# EXPOSE is set to 8080 to match Railway's Public Networking default (see
# the service's Settings -> Networking tab, which shows "-> Port 8080").
# Railway uses EXPOSE as a hint when auto-detecting which port to route
# traffic to and inject as $PORT; a mismatch between EXPOSE and the port
# Railway actually decided to route to can result in Railway's own edge
# proxy laying claim to that port inside the container's network namespace,
# which looks from inside the container like the port is permanently
# "unavailable" no matter what -- not a real process conflict, and not
# something retrying or picking a different port fixes. If you change the
# port in Railway's Networking settings, update this EXPOSE line (and the
# ENV default below) to match.
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PORT=8080
EXPOSE 8080

CMD ["python", "app/app.py"]