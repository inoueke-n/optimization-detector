FROM debian@sha256:a5edb9fa5b2a8d6665ed911466827734795ef10d2b3985a46b7e9c7f0161a6b3
RUN apt-get update
RUN apt-get -y install   \
    build-essential=12.6 \
    autoconf=2.69-11     \
    automake=1:1.16.1-4  \
    texinfo=6.5.0.dfsg.1-4+b1 \
    m4=1.4.18-2          \
    python3=3.7.3-1      \
    python3-pip=18.1-5   \
    cmake=3.13.4-1       \
    meson=0.49.2-1       \
    ninja-build=1.8.2-1
RUN pip3 install tqdm==4.53.0
RUN pip3 install python-magic==0.4.18
COPY resources /build/resources
COPY generate_dataset.py /build
WORKDIR /build
CMD ['bash']