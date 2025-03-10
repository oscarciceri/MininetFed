#!/bin/bash

sudo docker build -t "mininetfed:broker" -f docker/Dockerfile.broker .
sudo docker build -t "mininetfed:client" -f docker/Dockerfile.client .
sudo docker build -t "mininetfed:server" -f docker/Dockerfile.server .
sudo docker build -t "mininetfed:serversensor" -f docker/Dockerfile.serversensor .
sudo docker build -t "mininetfed:clientsensor" -f docker/Dockerfile.clientsensor .
