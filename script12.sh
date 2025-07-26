

REMOTE_PORT="$1"

# 传入参数1,2...  这样便于扩展，如果以后用了其他的frp，也好调整。

# frp execution binary only. 这里放frpc执行文件，如果版本有变也好改。配置文件模版，也在项目中, 模版文件名为template_frpc
wget -O  /kaggle/working/frp_0.54.0_linux_amd64.tar.gz https://github.com/fatedier/frp/releases/download/v0.54.0/frp_0.54.0_linux_amd64.tar.gz
tar -xzvf /kaggle/working/frp_0.54.0_linux_amd64.tar.gz -C /kaggle/working
cp -p /kaggle/working/frp_0.54.0_linux_amd64/frpc /kaggle/working/frpc
cp -p /kaggle/ComfyUI/template_frpc /kaggle/working/frpc.toml

FRP_CONFIG_FILE="/kaggle/working/frpc.toml"
CHOICE="$1"

echo $CHOICE

if [ "$CHOICE" -eq 1 ]; then
  TARGET_REMOTE_PORT="21663"
elif [ "$CHOICE" -eq 2 ]; then
  TARGET_REMOTE_PORT="21664"
else
  echo "Invalid CHOICE: $CHOICE"
  echo "Only 1 or 2 are supported."
  exit 1 # 退出并返回错误码
fi

echo "TARGET_REMOTE_PORT: $TARGET_REMOTE_PORT"

sed -i "s/REMOTE_PORT/$TARGET_REMOTE_PORT/g" "$FRP_CONFIG_FILE"
sleep 2
