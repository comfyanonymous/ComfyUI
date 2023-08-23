# Use the official Python image as the base image
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Run update
RUN apt update

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#Install Python

RUN apt-get install -y python3.9 
RUN apt-get install -y python3-pip
RUN apt-get install -y git

# Set the working directory inside the container
WORKDIR /app


# Copy the contents of the cloned GitHub repository to the working directory in the container
COPY ./ /app

# Install Python dependencies
RUN python3.9 -m pip install numpy==1.21.6
RUN python3.9 -m pip install -r requirements.txt
RUN python3.9 -m pip install scikit-learn==0.24.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install scikit-image
#RUN cd ./custom_nodes/comfy_controlnet_preprocessors && python3.9 install.py && cd ../
RUN cd ./custom_nodes/was-node-suite-comfyui/ && python3.9 -m pip install -r requirements.txt && cd ../
RUN cd ./custom_nodes/ComfyUI-Impact-Pack/ && python3.9 install.py && cd ../

#Give permission to script
RUN chmod +x ./entrypoint.sh

# Set the environment variable for GPU support
ENV NVIDIA_VISIBLE_DEVICES all

# Run the Python program when the container starts
ENTRYPOINT ["./entrypoint.sh"]
