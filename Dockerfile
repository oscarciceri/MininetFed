FROM ubuntu:xenial

# install required packages
RUN apt-get clean
RUN apt-get update \
    && apt-get install -y  git \
    net-tools \
    build-essential \
    python3-setuptools \
    python3-dev \
    python3-pip \
    software-properties-common \
    ansible \
    curl \
    iptables \
    iputils-ping \
    mosquitto \
    sudo

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
# install containernet (using its Ansible playbook)
RUN python3 -m pip install --upgrade "pip < 21.0"

# Hotfix: https://github.com/pytest-dev/pytest/issues/4770
RUN python3 -m pip install "more-itertools<=5.0.0"

# tell containernet that it runs in a container
ENV CONTAINERNET_NESTED 1



