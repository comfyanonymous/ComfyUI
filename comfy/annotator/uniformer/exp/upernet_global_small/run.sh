#!/usr/bin/env bash

work_path=$(dirname $0)
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train.py ${work_path}/config.py \
    --launcher pytorch \
    --options model.backbone.pretrained_path='your_model_path/uniformer_small_in1k.pth' \
    --work-dir ${work_path}/ckpt \
    2>&1 | tee -a ${work_path}/log.txt
