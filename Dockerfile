FROM ubuntu:18.04

ENV LANG=C.UTF-8
ENV TIME_ZONE=Asia/Shanghai

# 换源、安装软件、修改时区
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && apt-get update --fix-missing \
    && apt-get install -y wget tzdata \
    && apt-get install -y build-essential \
    && ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/{apt,dpkg,cache,log} /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/*

SHELL ["/bin/bash", "--login", "-c"]

# Create a non-root user
ARG username=user
ARG uid=1000
ARG gid=1000
ENV USER=$username
ENV UID=$uid
ENV GID=$gid
ENV HOME=/home/$USER
RUN addgroup --gid $GID $USER  && adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

COPY environment.yaml /tmp/
RUN chown $UID:$GID /tmp/environment.yaml

ENV APP_DIR=/app
ENV DATA_DIR=/data
ENV LOGS_DIR=/logs
RUN mkdir $APP_DIR && mkdir $DATA_DIR && mkdir $LOGS_DIR \
    && chown -R $UID $APP_DIR \
    && chown -R $UID $DATA_DIR \
    && chown -R $UID $LOGS_DIR
USER $USER

# install miniconda
ENV CONDA_DIR=$HOME/miniconda3
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
RUN conda config --add channels pytorch \
    && conda config --add channels conda-forge \
    && conda config --set remote_read_timeout_secs 1000.0 \
    && conda config --set remote_connect_timeout_secs 40 \
    && conda config --set show_channel_urls yes \
    && conda config --show \
    && echo "before update: $(conda --version)" \
    && conda update conda \
    && conda update --all \
    && echo "updated: $(conda --version)"

WORKDIR $APP_DIR

# pip换源
RUN PIP_EXISTS_ACTION=w pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && PIP_EXISTS_ACTION=w pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# build the conda environment
RUN conda create -n myconda python=3.11 \
    && conda init bash \
    && source activate \
    && conda activate myconda \
    && conda env update -n myconda --file /tmp/environment.yaml \
    && conda clean -p \
    && conda clean -t

# RUN PIP_EXISTS_ACTION=w conda clean -i && conda env create -n road --file /tmp/environment.yml --force \
#     && conda clean --all --yes \
#     && rm -rf $CONDA_DIR/pkgs/*

# 可以不用 activate the road env
# SHELL ["conda", "run", "-n", "road", "/bin/bash", "-c"]
## ADD CONDA ENV PATH TO LINUX PATH
ENV PATH=$CONDA_DIR/envs/myconda/bin:$PATH
#ENV CONDA_DEFAULT_ENV road

RUN echo "conda current env is: $(conda env list)"

# run the postBuild script to install the JupyterLab extensions
#RUN conda activate $ENV_PREFIX && \
#    /usr/local/bin/postBuild.sh && \
#    conda deactivate

COPY . .
# use an entrypoint script to insure conda environment is properly activated at runtime
#ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
CMD ["supervisord", "-c", "./supervisor/supervisord.conf"]
EXPOSE 5001
