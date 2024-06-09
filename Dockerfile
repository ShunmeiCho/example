# 使用NVIDIA的CUDA基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# 安装必要的软件包
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接，使 python 指向 python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# 安装Python包
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy numba

# 设置工作目录
WORKDIR /workspace

# 将当前目录的内容复制到工作目录中
COPY . /workspace

# 设置容器启动时执行的命令
CMD ["python"]
