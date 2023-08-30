FROM ubuntu:23.04

VOLUME /flw

# install required packages
RUN apt-get clean
RUN apt-get update \
    && apt-get install -y net-tools \
    curl \
    iptables \
    iputils-ping \
    mosquitto \
    sudo \
    python3.10 -y \
    python3-pip -y \
    python3-venv -y 
#     && python3 -m venv /flw/env

# RUN . /flw/env/bin/activate

# RUN pip3 install -r /flw/requirements.txt
# tell containernet that it runs in a container
ENV CONTAINERNET_NESTED 1



