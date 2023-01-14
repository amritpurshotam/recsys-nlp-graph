FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update -y
RUN apt-get -y install python3 \
    && apt-get -y install python3-pip

WORKDIR /app/

COPY requirements.txt /app/

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt