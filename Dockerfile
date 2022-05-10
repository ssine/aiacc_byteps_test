FROM nvidia/cuda:11.0-devel-ubuntu18.04

ARG https_proxy
ARG http_proxy

ARG BYTEPS_BASE_PATH=/usr/local
ARG BYTEPS_PATH=$BYTEPS_BASE_PATH/byteps
ARG BYTEPS_GIT_LINK=https://github.com/bytedance/byteps
ARG BYTEPS_BRANCH=master

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install -y software-properties-common wget \
    && add-apt-repository ppa:deadsnakes/ppa -y

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

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
        libcudnn8=8.4.0.*-1+cuda10.2 \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        python3.8 \
        python3.8-dev \
        python3.8-venv \
        libnccl2=2.12.10-1+cuda11.0 \
        libnccl-dev=2.12.10-1+cuda11.0

RUN python3.8 -m ensurepip --default-pip && pip3 install setuptools

# install framework
# note: for tf <= 1.14, you need gcc-4.9
ARG FRAMEWORK=pytorch
RUN if [ "$FRAMEWORK" = "tensorflow" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U tensorflow-gpu==1.15.0; \
    elif [ "$FRAMEWORK" = "pytorch" ]; then \
        pip3 install -U torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple; \
    elif [ "$FRAMEWORK" = "mxnet" ]; then \
        pip3 install -U mxnet-cu100==1.5.0; \
    else \
        echo "unknown framework: $FRAMEWORK"; \
        exit 1; \
    fi

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN cd $BYTEPS_BASE_PATH &&\
    git clone --recursive -b $BYTEPS_BRANCH $BYTEPS_GIT_LINK &&\
    cd $BYTEPS_PATH &&\
    python3.8 setup.py install

RUN apt-get install -y htop numactl && pip install wheel

RUN HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
