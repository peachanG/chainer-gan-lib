FROM nvidia/cuda:9.0-cudnn7-devel

COPY badproxy /etc/apt/apt.conf.d/99fixbadproxy

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS yes

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl git pkg-config apt-utils\
                                               python3 python3-dev python3-pip python3-setuptools cmake

RUN apt-get install -y libjpeg-dev libtiff-dev libjasper-dev libpng-dev \
                       libavcodec-dev libavformat-dev libswscale-dev \
                       libv4l-dev libatlas-base-dev gfortran libx264-dev

RUN pip3 install --upgrade pip
RUN pip3 install pandas h5py flask imagehash==3.4 matplotlib Pillow scikit-learn jupyter

WORKDIR /work

COPY . /work
RUN pip3 install -r requirements.txt

EXPOSE 8888
