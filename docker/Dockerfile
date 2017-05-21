#
# Docker file to run the BioIT 2017 workshop tutorial
#

FROM tensorflow/tensorflow:latest

MAINTAINER Wei-Yi Cheng

RUN apt-get update && apt-get install -y git-core && \
    rm -rf /var/apt/lists/*

RUN pip install keras h5py biopython scipy scikit-learn && \
    rm -rf /root/.cache/pip/*

ENV KERAS_BACKEND="tensorflow"

RUN useradd -ms /bin/bash bioit
USER bioit
WORKDIR /home/bioit

RUN git clone https://github.com/weiyi-bitw/bioitworld2017.git && \
    cd bioitworld2017 && \
    tar -zxvf bioitworld2017_data.tar.gz

