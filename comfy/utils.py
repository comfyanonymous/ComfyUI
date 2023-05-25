import torch
import math

def load_torch_file(ckpt, safe_load=False):
    if ckpt.lower().endswith(".safetensors"):
        import safetensors.torch
        sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def transformers_convert(sd, prefix_from, prefix_to, number):
    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}.transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}.transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd

#slow and inefficient, should be optimized
def bislerp(samples, width, height):
    shape = list(samples.shape)
    width_scale = (shape[3]) / (width )
    height_scale = (shape[2]) / (height )

    shape[3] = width
    shape[2] = height
    out1 = torch.empty(shape, dtype=samples.dtype, layout=samples.layout, device=samples.device)

    def algorithm(in1, in2, t):
        dims = in1.shape
        val = t

        #flatten to batches
        low = in1.reshape(dims[0], -1)
        high = in2.reshape(dims[0], -1)

        low_weight = torch.norm(low, dim=1, keepdim=True)
        low_weight[low_weight == 0] = 0.0000000001
        low_norm = low/low_weight
        high_weight = torch.norm(high, dim=1, keepdim=True)
        high_weight[high_weight == 0] = 0.0000000001
        high_norm = high/high_weight

        dot_prod = (low_norm*high_norm).sum(1)
        dot_prod[dot_prod > 0.9995] = 0.9995
        dot_prod[dot_prod < -0.9995] = -0.9995
        omega = torch.acos(dot_prod)
        so = torch.sin(omega)
        res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low_norm + (torch.sin(val*omega)/so).unsqueeze(1) * high_norm
        res *= (low_weight * (1.0-val) + high_weight * val)
        return res.reshape(dims)

    for x_dest in range(shape[3]):
        for y_dest in range(shape[2]):
            y = (y_dest + 0.5) * height_scale - 0.5
            x = (x_dest + 0.5) * width_scale - 0.5

            x1 = max(math.floor(x), 0)
            x2 = min(x1 + 1, samples.shape[3] - 1)
            wx = x - math.floor(x)

            y1 = max(math.floor(y), 0)
            y2 = min(y1 + 1, samples.shape[2] - 1)
            wy = y - math.floor(y)

            in1 = samples[:,:,y1,x1]
            in2 = samples[:,:,y1,x2]
            in3 = samples[:,:,y2,x1]
            in4 = samples[:,:,y2,x2]

            if (x1 == x2) and (y1 == y2):
                out_value = in1
            elif (x1 == x2):
                out_value = algorithm(in1, in3, wy)
            elif (y1 == y2):
                out_value = algorithm(in1, in2, wx)
            else:
                o1 = algorithm(in1, in2, wx)
                o2 = algorithm(in3, in4, wx)
                out_value = algorithm(o1, o2, wy)

            out1[:,:,y_dest,x_dest] = out_value
    return out1

def common_upscale(samples, width, height, upscale_method, crop):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples

        if upscale_method == "bislerp":
            return bislerp(s, width, height)
        else:
            return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))

@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, pbar = None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                        mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b:b+1] = out/out_div
    return output


PROGRESS_BAR_HOOK = None
def set_progress_bar_global_hook(function):
    global PROGRESS_BAR_HOOK
    PROGRESS_BAR_HOOK = function

class ProgressBar:
    def __init__(self, total):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK

    def update_absolute(self, value, total=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total)

    def update(self, value):
        self.update_absolute(self.current + value)
