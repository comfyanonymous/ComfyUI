# Use the official Python image as the base image
FROM nvidia/cuda:11.0.3-base-ubuntu20.04


# Run update
RUN apt update


#Install Python

RUN apt-get install -y python3 python3-pip


# Set the working directory inside the container
WORKDIR /app


# Copy the contents of the cloned GitHub repository to the working directory in the container
COPY ./ /app

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers
RUN pip3 install -r requirements.txt
RUN pip3 install runpod
RUN pip3 install websocket-client

#Give permission to script
RUN chmod +x ./entrypoint.sh

# Set the environment variable for GPU support
ENV NVIDIA_VISIBLE_DEVICES all

# Run the Python program when the container starts
ENTRYPOINT ["./entrypoint.sh"]
# [ "python3", "-u", "comfy_runpod.py" ]
