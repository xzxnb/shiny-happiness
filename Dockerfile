# Common

FROM --platform=amd64 nvidia/cuda:11.3.1-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && apt-add-repository ppa:swi-prolog/stable \
    && apt-get update \
    && apt-get install -y make gcc git curl \
    zlib1g zlib1g-dev python3-pip libssl-dev \
    bzip2 libbz2-dev sqlite3 libsqlite3-dev \
    libreadline-dev libffi-dev liblzma-dev \
    wget lmdb-utils python-dev build-essential \
    graphviz graphviz-dev cmake swi-prolog \
    texlive texlive-latex-extra  texlive-fonts-recommended \
    texlive-fonts-recommended texlive-generic-recommended texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-publishers  texlive-science  texlive-xetex \
    dvipng cm-super dvipng ghostscript cm-super

ARG USER_ID
ARG GROUP_ID
RUN (addgroup --gid $GROUP_ID app || true) \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID app

USER app

ENV HOME /home/app
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG PYTHON_VERSION

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pip install --upgrade pip \
    && pip install pipenv \
    && curl -sSL https://install.python-poetry.org | python -
ENV PATH="${PATH}:${HOME}/.local/bin"
ENV PATH $PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH

WORKDIR /app
ENV PYTHONPATH /app
ENV PYTHONDONTWRITEBYTECODE 1

# Standard stuff

RUN poetry config virtualenvs.create false

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN poetry install

# A100 compatible torch installation

RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pytorch-lightning==1.7.7
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# Some of DiGress stuff

USER root

RUN apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25 \
    && add-apt-repository "deb [ arch=amd64 ] https://downloads.skewed.de/apt bionic main" \
    && apt-get update \
    && apt-get install -y python3-graph-tool

USER app

RUN pip install git+https://github.com/igor-krawczuk/mini-moses@master

# End

ENV PYTHONPATH "/app:/app/DiGress"
