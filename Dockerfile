FROM microsoft/cntk:2.5-gpu-python3.5-cuda9.0-cudnn7.0

ARG git_owner="singnet"
ARG git_repo="time-series-decomposition"
ARG git_branch="master"

ENV SINGNET_REPOS=/opt/singnet
ENV SERVICE_NAME=time-series-decomposition

RUN mkdir -p ${SINGNET_REPOS}

RUN apt-get update && \
    apt-get install -y apt-utils

RUN apt-get install -y \
    curl \
    git \
    wget \
    nano

RUN SNETD_VERSION=`curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")'` && \
    cd /tmp && \
    wget https://github.com/singnet/snet-daemon/releases/download/${SNETD_VERSION}/snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    mv snet-daemon-${SNETD_VERSION}-linux-amd64/snetd /usr/bin/snetd

RUN cd ${SINGNET_REPOS} && \
    git clone -b ${git_branch} https://github.com/${git_owner}/${git_repo}.git

RUN cd ${SINGNET_REPOS}/${git_repo}/${SERVICE_NAME} && \
    /root/anaconda3/envs/cntk-py35/bin/python -m pip install -r requirements.txt && \
    sh buildproto.sh

WORKDIR ${SINGNET_REPOS}/${git_repo}/${SERVICE_NAME}
