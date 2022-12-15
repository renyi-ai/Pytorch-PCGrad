FROM pytorch/pytorch

# Install some basic apt packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    sudo vim nano git curl wget mc less \
    openssh-server \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-hun tesseract-ocr-deu \
    libicu-dev libpango1.0-dev libcairo2-dev \
    locales locales-all bc \
    ffmpeg libsm6 libxext6 enchant \
    g++ autoconf automake libtool pkg-config \
    libpng-dev  libjpeg8-dev  libtiff5-dev zlib1g-dev \
    libleptonica-dev myspell-hu \
    && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN pip install tqdm dotmap matplotlib==3.5.0 opencv-python pandas numpy  \
    jupyter sklearn tensorboard pudb pyenchant openpyxl tabulate xlrd fpdf scipy

ENV PYTHONPATH=/workspace
WORKDIR /workspace

RUN echo "%docker  ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/inside_sudoers

COPY entry.sh /entry.sh
RUN chmod +x /entry.sh
ENTRYPOINT ["/entry.sh"]
