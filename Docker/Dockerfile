FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update -y && apt-get upgrade -y &&\
  apt-get install vim git -y &&\
  apt-get autoremove -y &&\
  apt-get clean &&\
  rm -rf /usr/local/src/*
  
COPY requirements.txt requirements.txt
RUN pip install -U pip &&\
  pip install --no-cache-dir -r requirements.txt 
