FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir gradio audioldm2

COPY . .

ENV TOKENIZERS_PARALLELISM=true

EXPOSE 7860

CMD ["python3", "app.py"]