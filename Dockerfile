FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
# --- GNINA install (prebuilt Linux binary) ---
    RUN curl -L -o /tmp/gnina.tgz https://github.com/gnina/gnina/releases/download/v1.3.2/gnina-linux-x86_64-v1.3.2.tgz \
    && tar -C /usr/local/bin -xzvf /tmp/gnina.tgz gnina \
    && chmod +x /usr/local/bin/gnina \
    && rm -f /tmp/gnina.tgz
   
   # Meeko CLIs are installed from requirements (mk_prepare_* scripts)
   
