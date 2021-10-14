#!/bin/sh

docker build -t openvaccine ./Docker
docker run --gpus all -it -p 5555:5555 -v ${PWD}:/tf/${PWD##*/} -w /tf/${PWD##*/} openvaccine