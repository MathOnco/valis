FROM python:3.10

RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y gcc \
	g++ \
	libgeos-dev \
	openslide-tools \
	libvips-dev \
	maven \
	openjdk-11-jre

RUN pip install --no-cache --upgrade pip setuptools wheel

COPY valis valis
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY setup.cfg setup.cfg
RUN python3 -m pip install .

ENTRYPOINT ["python3"]
