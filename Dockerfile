FROM python:3.9.13-slim-bullseye

RUN apt-get update && apt-get install -y git
RUN mkdir /tmp/numba_cache && \
    chmod 777 /tmp/numba_cache

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN pip install numba==0.55.1 llvmlite
RUN pip install typing-extensions hyppo==0.3.0
RUN git clone https://github.com/microsoft/graspologic.git /graspologic && \
    cd /graspologic && \
    python setup.py install

RUN pip install ipython
