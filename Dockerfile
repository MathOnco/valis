FROM ubuntu:lunar

# Get build dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
	&& apt-get install -y \
		build-essential \
		software-properties-common \
		ninja-build \
		python3-pip \
		bc \
		wget 

# we need meson for libvips build
RUN pip3 install meson

# libvips dependencies
RUN apt-get install -y \
	glib-2.0-dev \
    openslide-tools \
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
    ca-certificates

# Openslide dependencies
RUN apt-get install -y \
	libc6 \
	libglib2.0-0 \
	libopenslide0 \
	libopenjp2-7-dev \
	libxml++2.6-dev \
	libsqlite3-dev


# Install Openslide and libvips
WORKDIR /usr/local/src

ENV LD_LIBRARY_PATH /usr/local/lib
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

ARG VIPS_VERSION=8.14.1
ARG VIPS_URL=https://github.com/libvips/libvips/releases/download

ARG OPENSLIDE_VERSION=3.4.1
ARG OPENSLIDE_URL=https://github.com/openslide/openslide/releases/download

RUN wget ${OPENSLIDE_URL}/v${OPENSLIDE_VERSION}/openslide-${OPENSLIDE_VERSION}.tar.gz --no-check-certificate \
	&& tar xf openslide-${OPENSLIDE_VERSION}.tar.gz \
	&& cd openslide-${OPENSLIDE_VERSION} \
	&& ./configure \
	&& make \
	&& make install 

RUN rm openslide-${OPENSLIDE_VERSION}.tar.gz

# build the head of the stable 8.14 branch
RUN wget ${VIPS_URL}/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz --no-check-certificate \
	&& tar xf vips-${VIPS_VERSION}.tar.xz \
	&& cd vips-${VIPS_VERSION} \
	&& meson build --buildtype=release --libdir=lib \
	&& cd build \
	&& ninja \
	&& ninja install

RUN rm vips-${VIPS_VERSION}.tar.xz

# Install other non-Python packages that are unrelated to libvips and openslide
RUN apt-get install -y maven \
    openjdk-11-jre

# Install python packages
COPY . .
RUN pip install --no-cache --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements_lock.txt

# Install valis-wsi source code
RUN python3 -m pip install .

# ENTRYPOINT ["python3"]