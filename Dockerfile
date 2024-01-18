FROM ubuntu:latest as builder

ARG WKDIR=/usr/local/src
WORKDIR ${WKDIR}

# ARG VIPS_VERSION=8.14.5
# ARG BF_VERSION=7.0.0

ARG VIPS_VERSION=8.15.1
ARG BF_VERSION=7.1.0
ARG PYTORCH_VERSION=2.0.1

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
RUN pip3 install meson
RUN apt-get install --no-install-recommends -y \
	glib-2.0-dev \
	libexpat-dev \
	librsvg2-dev \
	libpng-dev \
	libjpeg-turbo8-dev \
	libtiff-dev \
	libexif-dev \
	liblcms2-dev \
	libheif-dev \
	liborc-dev \
    libgirepository1.0-dev \
	libopenslide-dev


# RUN update-ca-certificates
# Install libvips from source to get latest version
ENV LD_LIBRARY_PATH /usr/local/lib
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

# build the head of the stable 8.14 branch
ARG VIPS_URL=https://github.com/libvips/libvips/releases/download
RUN wget ${VIPS_URL}/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz --no-check-certificate \
	&& tar xf vips-${VIPS_VERSION}.tar.xz \
	&& cd vips-${VIPS_VERSION} \
	&& meson build --buildtype=release --libdir=lib \
	&& cd build \
	&& ninja \
	&& ninja install

RUN rm vips-${VIPS_VERSION}.tar.xz
RUN rm -r vips-${VIPS_VERSION}


# Copy over necessary files
COPY valis valis
COPY pyproject.toml pyproject.toml
# COPY poetry.lock poetry.lock
COPY README.rst README.rst
COPY LICENSE.txt LICENSE.txt
COPY CITATION.cff CITATION.cff

# Install python packages using poetry
RUN pip3 install --upgrade pip
RUN pip3 install "poetry>=1.6.1"
RUN poetry config virtualenvs.in-project true

# RUN pip3 install poetry>=1.6.1
# RUN poetry config virtualenvs.in-project true

# RUN pip3 install poetry
# RUN poetry remove aicspylibczi
# Will install some Python packages with Git, so update git config
RUN git config --global http.sslVerify false
RUN poetry remove torch
RUN poetry lock
RUN poetry install --only main

# Set path to use .venv Python
ENV PATH="${WKDIR}/.venv/bin:$PATH"

RUN ${WKDIR}/.venv/bin/pip install --no-cache-dir torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

# # Install bioformats.jar in valis
RUN wget https://downloads.openmicroscopy.org/bio-formats/${BF_VERSION}/artifacts/bioformats_package.jar -P valis

# Clean up
RUN  apt-get remove -y wget build-essential ninja-build && \
  apt-get autoremove -y && \
  apt-get autoclean && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  rm -rf /usr/local/lib/python*


# # Copy over only what is needed to run, but not build, the package.
# # FROM ubuntu:jammy

# ARG WKDIR=/usr/local/src
# WORKDIR ${WKDIR}

# COPY --from=builder /usr/local/lib /usr/local/lib
# COPY --from=builder /etc/ssl/certs /etc/ssl/certs
# COPY --from=builder /usr/local/src /usr/local/src

# ENV LD_LIBRARY_PATH /usr/local/lib
# ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig
# ENV PATH="${WKDIR}/.venv/bin:$PATH"

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update \
# 	&& apt-get install -y \
# 	glib-2.0-dev \
# 	libexpat-dev \
# 	librsvg2-dev \
# 	libpng-dev \
# 	libjpeg-turbo8-dev \
# 	libtiff-dev \
# 	libexif-dev \
# 	liblcms2-dev \
# 	libheif-dev \
# 	liborc-dev \
# 	libgirepository1.0-dev \
# 	libopenslide-dev \
# 	libjxr-dev \
# 	openjdk-11-jre

# Install other non-Python dependencies
# RUN apt-get install -y \
    # openjdk-11-jre
