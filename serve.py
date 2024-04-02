import os

COMFY_PORT = 8188
LOCALHOST = '0.0.0.0'
OUTPUT_DIR = '/opt/ml/code/efs'


cmd = "python main.py  --listen {} --port {} --output-directory {}".format(LOCALHOST, COMFY_PORT, OUTPUT_DIR)
os.system(cmd)

