#!/usr/bin/env bash

work_path=$(dirname $0)
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/test.py ${work_path}/test_config_h32.py \
    ${work_path}/ckpt/latest.pth \
    --launcher pytorch \
    --eval mIoU \
    2>&1 | tee -a ${work_path}/log.txt
