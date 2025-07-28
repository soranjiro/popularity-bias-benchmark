# ベースイメージとしてCUDA対応のMinicondaイメージを使用
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# Minicondaのインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 環境変数の設定
ENV PATH /opt/conda/bin:$PATH

# environment.yamlをコピー
COPY environment.yaml .

# Condaの利用規約を受諾
RUN conda config --set notify_outdated_conda false

# 必要なチャンネルの利用規約を受諾
RUN conda config --add channels conda-forge
RUN conda config --add channels defaults
RUN conda config --add channels pytorch
RUN /bin/bash -c "echo 'y' | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main"
RUN /bin/bash -c "echo 'y' | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"

# conda環境をenvironment.yamlから作成
RUN /bin/bash -c "conda env create -f environment.yaml"

# run code
CMD ["/bin/bash", "-c", "source /opt/conda/bin/activate myenv && python main.py"]
