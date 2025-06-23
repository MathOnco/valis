FROM ubuntu:noble AS builder

ARG WKDIR=/usr/local/src
WORKDIR ${WKDIR}

ARG UV_VERSION=0.6.5
ARG VIPS_VERSION=8.16.0
ARG BF_VERSION=7.0.0
ARG PYTORCH_VERSION=2.4.0
ARG TORCHVISION_VERSION=0.20.1
ARG OPENCV_VERSION=4.9.0.80

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Get build dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
	&& apt-get install -y \
		build-essential \
		software-properties-common \
		ninja-build \
		python3-pip \
		bc \
		wget \
		ca-certificates \
		git-all \
		cmake \
		libjxr-dev \
    openjdk-11-jre

# libvips dependencies for libvips build
RUN apt-get install -y python3-venv
RUN python3 -m venv ~/.local
RUN echo $PATH
ENV PATH="$PATH:~/.local/bin"
RUN echo $PATH
RUN ~/.local/bin/pip3 install meson

RUN apt-get update
RUN apt-get install --no-install-recommends -y \
	libglib2.0-dev \
	glib-2.0-dev \
	libexpat1-dev \
	libexpat-dev \
	librsvg2-2 \
	librsvg2-common \
	librsvg2-dev \
	libpng-dev \
	libjpeg-turbo8-dev \
	libopenjp2-7-dev \
	libtiff-dev \
	libexif-dev \
	liblcms2-dev \
	libheif-dev \
	liborc-dev \
  	libgirepository1.0-dev \
	libopenslide-dev \
	librsvg2-dev

RUN update-ca-certificates
### Install libvips from source to get latest version
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

# build the head of the stable 8.14 branch
ARG VIPS_URL=https://github.com/libvips/libvips/releases/download
RUN wget ${VIPS_URL}/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz --no-check-certificate \
	&& tar xf vips-${VIPS_VERSION}.tar.xz \
	&& cd vips-${VIPS_VERSION} \
	&& ~/.local/bin/meson build --buildtype=release --libdir=lib \
	&& cd build \
	&& ninja \
	&& ninja install

RUN rm vips-${VIPS_VERSION}.tar.xz
RUN rm -r vips-${VIPS_VERSION}

# Copy over necessary files
COPY valis valis
COPY pyproject.toml pyproject.toml
COPY README.rst README.rst
COPY LICENSE.txt LICENSE.txt
COPY CITATION.cff CITATION.cff

# Install python packages using UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# RUN uv sync
RUN uv pip install .
# Set path to use .venv Python
ENV PATH="${WKDIR}/.venv/bin:$PATH"

# Install bioformats.jar in valis
RUN wget https://downloads.openmicroscopy.org/bio-formats/${BF_VERSION}/artifacts/bioformats_package.jar -P valis

# Download pytorch model weights
COPY ./docker/docker_download_weights.py docker_download_weights.py
RUN python3 docker_download_weights.py

# Clean up
RUN apt-get remove -y wget build-essential ninja-build && \
  apt-get autoremove -y && \
  apt-get autoclean && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  rm -rf /usr/local/lib/python*

