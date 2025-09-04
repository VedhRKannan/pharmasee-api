FROM python:3.11-slim

# install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git wget curl \
    libeigen3-dev \
    openbabel \
 && rm -rf /var/lib/apt/lists/*

# clone and build smina from source
RUN git clone https://github.com/mwojcikowski/smina.git /opt/smina \
 && cd /opt/smina \
 && mkdir build && cd build \
 && cmake .. \
 && make -j$(nproc) \
 && cp smina /usr/local/bin/smina

# python deps
RUN pip install flask rdkit-pypi joblib numpy biopython

WORKDIR /app
COPY . /app

CMD ["python", "app.py"]
