FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install system deps needed for RDKit to run
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install flask rdkit-pypi joblib numpy biopython

WORKDIR /app
COPY . /app

EXPOSE 8080
CMD ["python", "app.py"]
