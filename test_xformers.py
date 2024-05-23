# ref :https://github.com/facebookresearch/xformers/issues/845


import xformers
import xformers.ops
import torch,os
os.environ["MAX_BLOCK_SIZE"] = "655350"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

q = torch.zeros(([65536, 16, 80])).cuda()
k = torch.zeros(([65536, 16, 80])).cuda()
v = torch.zeros(([65536, 16, 80])).cuda()
out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

print("Done.")