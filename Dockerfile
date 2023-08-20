FROM ubuntu:18.04

# Set the shell to be used
SHELL ["/bin/bash", "-c"]

# Update package lists and remove cached package lists
RUN set -xe && \
    rm -rf /var/lib/apt/lists/*

# Create a directory for SMOKE project and KITTI dataset
RUN mkdir -p /soe/SMOKE
RUN mkdir -p /soe/kitti

COPY Documents/PycharmProjects/SMOKE /soe/SMOKE
COPY Documents/PycharmProjects/kitti /soe/kitti



# Set the default command when running the container
CMD ["/bin/bash"]

# Set the maintainer label
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"


# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA=cuda>=10.0 brand=tesla,driver>=410,driver<411

# Set environment variables for NCCL
ENV NCCL_VERSION=2.4.2

# Set environment variable for MKL-DNN
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# Set environment variables for CUDNN
ENV CUDNN_VERSION=7.4.2.24


# Start sshd daemon
CMD ["/usr/sbin/sshd", "-D"]