FROM buildpack-deps:focal as buildermaster
LABEL maintainer="s.graber@fz-juelich.de"

## Modified to include:
# pydmd

SHELL ["/bin/bash", "-c"]

ARG WITH_MPI=ON
ARG WITH_OMP=ON
ARG WITH_GSL=ON
#ARG WITH_MUSIC=ON
ARG WITH_LIBNEUROSIM=OFF

ENV TERM=xterm \
    TZ=Europe/Berlin \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libtool automake autotools-dev libreadline8 libreadline-dev freeglut3-dev \
    gosu \
    cmake \
    cython3 \
    jq \
    libboost1.67-dev \
    libgomp1 \
    libgsl-dev \
    libltdl7 \
    libltdl-dev \
    libmusic1v5 \
    libopenmpi-dev \
    libomp-dev \
    libpcre3 \
    libpcre3-dev \
    libpython3.8 \
    llvm-dev \
    openmpi-bin \
    pandoc \
    pep8 \
    python3-dev \
    python3-ipython \
    python3-jupyter-core \
    python3-matplotlib \
    python3-mpi4py \
    python3-nose \
    python3-numpy \
    python3-pandas \
    python3-path \
    python3-pip \
    python3-scipy \
    python3-setuptools \
    python3-statsmodels \
    python3-tk \
    python-dev \
    bison \
    flex \
    libhdf5-dev \
    vera++ \
    wget  && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    #update-alternatives --remove-all python && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    python3 -m pip install --upgrade pip setuptools wheel


## Install Neuron



## Install NEST
RUN git clone https://github.com/nest/nest-simulator.git && \
  cd nest-simulator && \
  git checkout master && \
  python3 -m pip install -r ./doc/requirements.txt && \
  cd .. && \
  mkdir nest-build && \
  ls -l && \
  cd  nest-build && \
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/nest/ \
        # -Dwith-optimize=ON \
        # -Dwith-warning=ON \
        -Dwith-ltdl=ON \
        -Dwith-gsl=$WITH_GSL \
        -Dwith-readline=ON \
        -Dwith-python=ON \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
        -Dwith-mpi=$WITH_MPI \
        -Dwith-openmp=$WITH_OMP \
        -Dwith-libneurosim=$WITH_LIBNEUROSIM \
#       -Dwith-music=/opt/music-install \
        ../nest-simulator && \
  make && \
  pip install setuptools==58.2.0 && \
  source bin/nest_vars.sh && \
  make install

# Add NEST binary folder to PATH
RUN echo "source /opt/nest/bin/nest_vars.sh" >> root/.bashrc

## Install Neuron
RUN git clone --depth 1 -b 8.0.0 https://github.com/neuronsimulator/nrn.git /usr/src/nrn
RUN mkdir nrn-bld

RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/nrn/ \
  -DCURSES_NEED_NCURSES=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_RX3D=OFF \
  -DNRN_ENABLE_PYTHON=ON \
  -S /usr/src/nrn \
  -B nrn-bld

RUN cmake --build nrn-bld --parallel 4 --target install

# add nrnpython to PYTHONPATH
ENV PYTHONPATH /opt/nrn/lib/python:${PYTHONPATH}

# clean up
RUN rm -r /usr/src/nrn
RUN rm -r nrn-bld

# ---- pip install some additional things
RUN pip install pymoo
RUN pip install git+https://github.com/NeuralEnsemble/parameters

# ---- install NESTML -----
RUN pip install antlr4-python3-runtime
RUN pip install git+https://github.com/nest/nestml.git

# ---- install LFPykernels (main branch) -----
RUN pip install git+https://github.com/LFPy/LFPykernels



###############################################################################

FROM ubuntu:focal
LABEL maintainer="s.graber@fz-juelich.de"

ENV TERM=xterm \
    TZ=Europe/Berlin \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpcre3 \
        libpcre3-dev \
        gosu \
        jupyter-notebook \
        less \
        libgomp1 \
        libgsl-dev  \
        libltdl7 \
        libopenmpi-dev \
        libomp-dev \
        libpython3.8 \
        nano \
        openmpi-bin \
        openssh-client \
        openssh-server \
        python3-dev \
        python3-flask \
        python3-flask-cors \
        python3-restrictedpython \
        python3-matplotlib \
        python3-mpi4py \
        python-numpy \
        python3-pip \
        python3-scipy \
        python3-setuptools \
        python3-pandas \
        python3-sympy \
        python3-tk \
        wget  && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    pip install quantities lazyarray neo pydmd bayesian-optimization mpi4py && \
    pip install uwsgi &&\
     wget https://github.com/NeuralEnsemble/PyNN/archive/nest-dev.tar.gz && \
     tar -xzf nest-dev.tar.gz && \
     cd PyNN-nest-dev && \
     python3 setup.py install && \
     cd .. && rm -rf PyNN-nest-dev && rm nest-dev.tar.gz && \
    #pip install --no-binary :all: PyNN
     pip install numpy==1.23.1 && \
     mkdir code && \
     mkdir data

COPY --from=buildermaster /opt/nest /opt/nest

COPY . /code

EXPOSE 5000 8080

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && \
        chmod +x /code/simulation/run.py


ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
#ENTRYPOINT ["/bin/bash"]
