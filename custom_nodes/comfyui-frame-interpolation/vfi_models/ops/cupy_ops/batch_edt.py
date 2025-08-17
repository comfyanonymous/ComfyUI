############### DISTANCE TRANSFORM ###############
# img tensor: (bs,h,w) or (bs,1,h,w)
# returns same shape
# expects white lines, black whitespace
# defaults to diameter if empty image
from .utils import cuda_kernel, cuda_launch, cuda_int32, cuda_float32
import torch

_batch_edt_kernel = (
    "kernel_dt",
    """
    extern "C" __global__ void kernel_dt(
        const int bs,
        const int h,
        const int w,
        const float diam2,
        float* data,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= bs*h*w) {
            return;
        }
        int pb = idx / (h*w);
        int pi = (idx - h*w*pb) / w;
        int pj = (idx - h*w*pb - w*pi);

        float cost;
        float mincost = diam2;
        for (int j = 0; j < w; j++) {
            cost = data[h*w*pb + w*pi + j] + (pj-j)*(pj-j);
            if (cost < mincost) {
                mincost = cost;
            }
        }
        output[idx] = mincost;
        return;
    }
""",
)
_batch_edt = None


def batch_edt(img, block=1024):
    # must initialize cuda/cupy after forking
    global _batch_edt
    if _batch_edt is None:
        _batch_edt = cuda_launch(*_batch_edt_kernel)

    # bookkeeppingg
    if len(img.shape) == 4:
        assert img.shape[1] == 1
        img = img.squeeze(1)
        expand = True
    else:
        expand = False
    bs, h, w = img.shape
    diam2 = h**2 + w**2
    odtype = img.dtype
    grid = (img.nelement() + block - 1) // block

    # cupy implementation
    if img.is_cuda:
        # first pass, y-axis
        data = ((1 - img.type(torch.float32)) * diam2).contiguous()
        intermed = torch.zeros_like(data)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),  # < 1024
            args=[
                cuda_int32(bs),
                cuda_int32(h),
                cuda_int32(w),
                cuda_float32(diam2),
                data.data_ptr(),
                intermed.data_ptr(),
            ],
        )

        # second pass, x-axis
        intermed = intermed.permute(0, 2, 1).contiguous()
        out = torch.zeros_like(intermed)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            args=[
                cuda_int32(bs),
                cuda_int32(w),
                cuda_int32(h),
                cuda_float32(diam2),
                intermed.data_ptr(),
                out.data_ptr(),
            ],
        )
        ans = out.permute(0, 2, 1).sqrt()
        ans = ans.type(odtype) if odtype != ans.dtype else ans

    # default to scipy cpu implementation
    else:
        raise NotImplementedError()
        """ sums = img.sum(dim=(1, 2))
        ans = torch.tensor(
            np.stack(
                [
                    scipy.ndimage.morphology.distance_transform_edt(i)
                    if s != 0
                    else np.ones_like(i)  # change scipy behavior for empty image
                    * np.sqrt(diam2)
                    for i, s in zip(1 - img, sums)
                ]
            ),
            dtype=odtype,
        ) """

    if expand:
        ans = ans.unsqueeze(1)
    return ans

__all__ = ["batch_edt"]