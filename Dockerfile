FROM ubuntu@sha256:3093096ee188f8ff4531949b8f6115af4747ec1c58858c091c8cb4579c39cc4e
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -yq install            \
    build-essential=12.8ubuntu1   \
    flex=2.6.4-6.2                \
    bison=2:3.5.1+dfsg-1          \
    autoconf=2.69-11.1            \
    automake=1:1.16.1-4ubuntu6    \
    texinfo=6.7.0.dfsg.2-5        \
    m4=1.4.18-4                   \
    python3=3.8.2-0ubuntu2        \
    python3-pip=20.0.2-5ubuntu1.1 \
    cmake=3.16.3-1ubuntu1         \
    meson=0.53.2-2ubuntu2         \
    ninja-build=1.10.0-1build1    \
    libtool=2.4.6-14              \
    pkg-config=0.29.1-0ubuntu4    \
    gawk=1:5.0.1+dfsg-1           \
    file=1:5.38-4

RUN pip3 install tqdm==4.53.0
RUN pip3 install python-magic==0.4.18

COPY resources /build/resources
COPY generate_dataset.py /build
WORKDIR /build
CMD ['bash']
