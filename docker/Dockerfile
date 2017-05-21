#
# Docker file to build container with tensorflow, Keras
#

FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Wei-Yi Cheng

RUN apt-get update && apt-get install -y git-core && \
    rm -rf /var/apt/lists/*

RUN pip install keras pandas h5py && \
    rm -rf /root/.cache/pip/*

RUN cd / && git clone  && \
    cd bioitworld2017 && \
    tar -zxvf bioitworld2017_data.tar.gz

