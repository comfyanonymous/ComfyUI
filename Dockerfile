FROM python:3.12-slim
EXPOSE 8188

# Copy the repo and install required dependencies
WORKDIR /ComfyUI
COPY . .
RUN apt update && apt install git -y
RUN pip install --no-cache-dir -r requirements.txt

# ComfyUI entrypoint
WORKDIR /ComfyUI
CMD [ "python", "main.py" ]