# Use slim Python base
FROM python:3.11-slim

# Install system deps for RDKit & docking tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install smina (static build)
RUN curl -L -o /tmp/smina.static https://github.com/ccsb-scripps/smina/releases/download/rel-2020-12-10/smina.static \
 && chmod +x /tmp/smina.static \
 && mv /tmp/smina.static /usr/local/bin/smina

# Copy app code
COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
