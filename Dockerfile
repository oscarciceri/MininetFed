FROM ubuntu:22.04

VOLUME /flw

# install required packages
RUN apt-get clean

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

RUN apt-get update \
    && apt-get install -y net-tools \
    curl \
    iptables \
    iputils-ping \
    mosquitto \
    sudo \
    # python3.10 -y \
    python3-pip -y \
    python3-venv -y 
#     && python3 -m venv /flw/env

# Ports used by Mosquitto host:
EXPOSE 1883
EXPOSE 8883
# RUN . /flw/env/bin/activate

# RUN pip3 install -r /flw/requirements.txt
# tell containernet that it runs in a container
ENV CONTAINERNET_NESTED 1



