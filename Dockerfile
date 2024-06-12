FROM tensorflow/tensorflow:latest-gpu

WORKDIR /src

COPY ./env .

RUN apt update -y && pip install -r requirements.txt && rm -rf ./env