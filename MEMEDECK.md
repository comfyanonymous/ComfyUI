# Commands to set up the environment.

## Enable MIGs on NVIDA GPU

<!-- link to the guide -->
https://www.seimaxim.com/kb/gpu/nvidia-a100-mig-cheat-sheat

```zsh  
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -cgi 9,9 -C
```

Start comfy with MIGs

```zsh
CUDA_VISIBLE_DEVICES=MIG-0 python main.py --port 5000 --listen 0.0.0.0 --cuda-device 0 --preview-method auto
```


## Tunnel remote server port to local machine port

### On the server

```zsh
sudo nano /etc/ssh/sshd_config

# uncomment this line in the sshd_config file
GatewayPorts yes
```
Then restart the machine.

## On your local machine
```zsh
sudo ssh -i ~/.ssh/memedeck-monolith.pem -N -R 9090:localhost:8079 holium@172.206.15.40 
```

## install autossh (maybe not needed)
```zsh
brew install autossh
autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -i ~/.ssh/memedeck-monolith.pem -N -R 9090:localhost:8079 holium@172.206.15.40
```
<!-- sudo ssh -i ~/.ssh/memedeck-monolith.pem -N -R 9090:localhost:8079 holium@172.206.15.40 -->


## On the server
This is to allow the port to be accessed from the local machine.

```zsh
sudo iptables -A INPUT -p tcp --dport 9090 -j ACCEPT
```


### Quick start for dev

```zsh
# ssh into to the server
ssh -i ~/.ssh/memedeck-monolith.pem holium@172.206.15.40

# on a100 server
API_ADDRESS='http://localhost:9090/v2' API_KEY='<your-api-key>' AMQP_ADDR='amqp://api:gacdownatravKekmy9@51.8.120.154:5672/dev' python main.py --port 5001 --listen 0.0.0.0 --cuda-device 0 --preview-method auto

# on local machine
ssh -N -R 9090:0.0.0.0:8079 -i ~/.ssh/memedeck-monolith.pem holium@172.206.15.40 
```