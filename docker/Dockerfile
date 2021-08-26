FROM nvidia/cuda:11.1-devel-ubuntu18.04

ARG https_proxy
ARG http_proxy

ARG BYTEPS_BASE_PATH=/usr/local
ARG BYTEPS_PATH=$BYTEPS_BASE_PATH/byteps
ARG BYTEPS_GIT_LINK=https://github.com/xiezhq-hermann/byteps
ARG BYTEPS_BRANCH=master

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    tzdata \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    cmake \
    lsb-release \
    libcudnn8=8.0.5.39-1+cuda11.1 \
    libnuma-dev \
    ibverbs-providers \
    librdmacm-dev \
    ibverbs-utils \
    rdmacm-utils \
    libibverbs-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libnccl2=2.8.4-1+cuda11.1 \
    libnccl-dev=2.8.4-1+cuda11.1

# install framework
# note: for tf <= 1.14, you need gcc-4.9
ARG FRAMEWORK=tensorflow
RUN if [ "$FRAMEWORK" = "pytorch" ]; then \
    # for YOLOV3
    pip3 install -U numpy xlsxwriter\
    torchvision==0.10.0+cu111 torch==1.9.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html; \
    # elif [ "$FRAMEWORK" = "tensorflow" ]; then \
    #     pip3 install --upgrade pip; \
    #     pip3 install -U tensorflow-gpu==1.15.0; \
    # elif [ "$FRAMEWORK" = "mxnet" ]; then \
    #     pip3 install -U mxnet-cu100==1.5.0; \
    else \
    echo "unknown framework: $FRAMEWORK"; \
    exit 1; \
    fi

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN cd $BYTEPS_BASE_PATH &&\
    git clone --recursive -b $BYTEPS_BRANCH $BYTEPS_GIT_LINK &&\
    cd $BYTEPS_PATH &&\
    python3 setup.py install

####### to compile pytorch from source:
# install conda
# conda install -c pytorch magma-cuda111
# RUN git clone https://github.com/xiezhq-hermann/pytorch.git /root/pytorch --recursive --branch release/1.9
# cd pytorch
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# python setup.py install
