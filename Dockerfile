FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt update -y && \
    apt upgrade -y
RUN apt install -y --no-install-recommends \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/local/bin/pip pip /usr/bin/pip3 10

ENV NB_USER user
ENV NB_UID 1001

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER 

RUN pip install --user setuptools wheel && \
    pip install --user chainer --pre cupy && \
    pip install scikit-learn scipy pandas tqdm autopep8

ENV CHAINER_DATASET_ROOT /src/.dataset

RUN pip install --user nltk && \
    python -m nltk.downloader punkt

RUN pip install --user jupyter
ENV PATH /home/user/.local/bin:$PATH
RUN jupyter nbextension enable --py widgetsnbextension && \
    jupyter nbextension install https://github.com/kenkoooo/jupyter-autopep8/archive/master.zip --user && \
    jupyter nbextension enable jupyter-autopep8-master/jupyter-autopep8

WORKDIR /src

CMD jupyter notebook --ip 0.0.0.0 --port 8888
