FROM ubuntu:focal
#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y
RUN apt-get install nano
RUN apt-get install -y git
RUN apt-get install -y python3.8 python3-pip

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libopencv-dev
 
#RUN pip --default-timeout=1000 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install tensorboard
RUN pip install opencv-python
RUN pip install scipy
RUN pip install numpy
RUN pip install matplotlib
#RUN pip install pyyaml
#RUN pip install torchtyping
#RUN pip install setuptools
#RUN pip install protobuf
RUN pip install tqdm
RUN pip install flake8

WORKDIR /work
