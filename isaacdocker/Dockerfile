FROM ubuntu:22.04

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    sudo \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-glx \
    libxrandr2 \
    libegl1 \
    libvulkan1 \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with sudo privileges
RUN useradd -m -s /bin/bash isaac && \
    echo "isaac ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the non-root user
USER isaac
WORKDIR /home/isaac

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh

# Add conda to path and initialize
ENV PATH="/home/isaac/miniconda/bin:${PATH}"
RUN conda init bash && \
    echo "conda activate isaac" >> ~/.bashrc

# Create a conda environment for Isaac Lab with Python 3.10
RUN conda create -n isaac python=3.10 -y

# Install PyTorch and Isaac Sim in the conda environment
RUN /bin/bash -c "source ~/miniconda/bin/activate isaac && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com"

# Create workspace directory
RUN mkdir -p /home/isaac/workspace

# Create a simple entrypoint script that just activates conda
RUN echo '#!/bin/bash\n\
source ~/miniconda/bin/activate isaac\n\
exec "$@"\n' > /home/isaac/entrypoint.sh && \
    chmod +x /home/isaac/entrypoint.sh

# Set the working directory
WORKDIR /home/isaac/workspace

# Set the entrypoint
ENTRYPOINT ["/home/isaac/entrypoint.sh"]
CMD ["/bin/bash"]
