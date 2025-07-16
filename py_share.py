import threading
import time
import subprocess

import os
from pyngrok import ngrok, conf

# Replace 'YOUR_NGROK_AUTH_TOKEN' with your actual token
# This helps avoid connection limits for unauthenticated users.
conf.get_default().auth_token = "2mpZeIQCMUw0OBOVtSsfDlhtmBH_2jXryAAHrdL1tLRaaPkSA"

def ngrok_thread(port):
    try:
        # ngrok will try to tunnel to port 5000
        public_url = ngrok.connect(addr=port)
        print(f"ngrok tunnel established at: {public_url}")
    except Exception as e:
        print(f"Error starting ngrok tunnel: {e}")
        print("This often happens due to Kaggle's network restrictions.")




from zrok.proxy import ProxyShare
import zrok

def zrok_thread(port):
    # Load the user's zrok environment from ~/.zrok
    zrok_env = zrok.environment.root.Load()
    
    # Create a temporary proxy share (will be cleaned up on exit) 运行时停止的时候自动清理
    proxy = ProxyShare.create(root=zrok_env, target=f"http://127.0.0.1:{port}") #
    
    print(f"Public URL: {proxy.endpoints}")
    proxy.run()


# threading.Thread(target=ngrok_thread, daemon=True, args=(8188,)).start()
# threading.Thread(target=zrok_thread, daemon=True, args=(8188,)).start()
# time.sleep(5)  # 观察5秒内是否有输出
