FROM ubuntu:18.04

ENV LANG=C.UTF-8
ENV TIME_ZONE=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装必要工具
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
        wget \
        curl \
        tar \
        vim \
        build-essential \
        tzdata && \
    ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && \
    echo $TIME_ZONE > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建非root用户
ARG username=user
ARG uid=1000
ARG gid=1000
ENV USER=$username
ENV UID=$uid
ENV GID=$gid
ENV HOME=/home/$USER

RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID --create-home --shell /bin/bash $USER

# 创建应用目录
ENV APP_DIR=/app
ENV DATA_DIR=/data
ENV LOGS_DIR=/logs

RUN mkdir -p $APP_DIR $DATA_DIR $LOGS_DIR && \
    chown -R $UID:$GID $APP_DIR $DATA_DIR $LOGS_DIR

# 切换到非root用户
USER $USER
WORKDIR $APP_DIR

# 安装Miniconda
ENV CONDA_DIR=$HOME/miniconda3
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# 配置conda环境
ENV PATH=$CONDA_DIR/bin:$PATH

# 配置pip镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 复制环境配置文件
COPY --chown=$UID:$GID environment.yaml /tmp/

# 创建conda环境
RUN conda create -n myconda python=3.11 -y && \
    conda env update -n myconda --file /tmp/environment.yaml && \
    conda clean --all -y

# 设置conda环境路径
ENV PATH=$CONDA_DIR/envs/myconda/bin:$PATH

# 复制应用代码
COPY --chown=$UID:$GID . .

# 暴露端口
EXPOSE 5088

# 启动命令
CMD ["supervisord", "-c", "./flask_api/supervisor/supervisord.conf"]
