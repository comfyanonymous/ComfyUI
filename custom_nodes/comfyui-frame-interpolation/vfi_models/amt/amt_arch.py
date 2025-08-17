"""
https://github.com/MCG-NKU/AMT/blob/main/utils/dist_utils.py
https://github.com/MCG-NKU/AMT/blob/main/utils/flow_utils.py
https://github.com/MCG-NKU/AMT/blob/main/utils/utils.py
https://github.com/MCG-NKU/AMT/blob/main/networks/blocks/feat_enc.py
https://github.com/MCG-NKU/AMT/blob/main/networks/blocks/ifrnet.py
https://github.com/MCG-NKU/AMT/blob/main/networks/blocks/multi_flow.py
https://github.com/MCG-NKU/AMT/blob/main/networks/blocks/raft.py
https://github.com/MCG-NKU/AMT/blob/main/networks/AMT-S.py
https://github.com/MCG-NKU/AMT/blob/main/networks/AMT-L.py
https://github.com/MCG-NKU/AMT/blob/main/networks/AMT-G.py
"""
#Removed imageio by removing readImage, writeImage
#The model will receive image tensors from other ComfyUI's nodes so they are unneccessary

import torch
import torch.nn as nn
import numpy as np
from PIL import ImageFile
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
import sys
import random

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)











class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterGroups:
    def __init__(self) -> None:
        self.meter_dict = dict()
    
    def update(self, dict, n=1):
        for name, val in dict.items():
            if self.meter_dict.get(name) is None:
                self.meter_dict[name] = AverageMeter()
            self.meter_dict[name].update(val, n)
    
    def reset(self, name=None):
        if name is None:
            for v in self.meter_dict.values():
                v.reset()
        else:
            meter = self.meter_dict.get(name)
            if meter is not None:
                meter.reset()
    
    def avg(self, name):
        meter = self.meter_dict.get(name)
        if meter is not None:
            return meter.avg


class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, input_tensor):
        return F.pad(input_tensor, self._pad, mode='replicate')

    def unpad(self, input_tensor):
        return self._unpad(input_tensor)
    
    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def img2tensor(img):
    if img.shape[-1] > 3:
        img = img[:,:,:3]
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0


def tensor2img(img_t):
    return (img_t * 255.).detach(
                        ).squeeze(0).permute(1, 2, 0).cpu().numpy(
                        ).clip(0, 255).astype(np.uint8)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)


def check_dim_and_resize(tensor_list):
    shape_list = []
    for t in tensor_list:
        shape_list.append(t.shape[2:])

    if len(set(shape_list)) > 1:
        desired_shape = shape_list[0]
        print(f'Inconsistent size of input video frames. All frames will be resized to {desired_shape}')
        
        resize_tensor_list = []
        for t in tensor_list:
            resize_tensor_list.append(torch.nn.functional.interpolate(t, size=tuple(desired_shape), mode='bilinear'))

        tensor_list = resize_tensor_list

    return tensor_list











class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(72, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

class LargeEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(LargeEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(112, stride=2)
        self.layer3 = self._make_layer(160, stride=2)
        self.layer3_2 = self._make_layer(160, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(self.in_planes, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_2(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x











def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        res_feat = out[:, :-self.side_channels, ...]
        side_feat = out[:, -self.side_channels:, :, :]
        side_feat = self.conv2(side_feat)
        out = self.conv3(torch.cat([res_feat, side_feat], 1))

        res_feat = out[:, :-self.side_channels, ...]
        side_feat = out[:, -self.side_channels:, :, :]
        side_feat = self.conv4(side_feat)
        out = self.conv5(torch.cat([res_feat, side_feat], 1))

        out = self.prelu(x + out)
        return out
    
class Encoder(nn.Module):
    def __init__(self, channels, large=False):
        super(Encoder, self).__init__()
        self.channels = channels        
        prev_ch = 3
        for idx, ch in enumerate(channels, 1):
            k = 7 if large and idx == 1 else 3
            p = 3 if k ==7 else 1
            self.register_module(f'pyramid{idx}', 
            nn.Sequential(
                convrelu(prev_ch, ch, k, 2, p),
                convrelu(ch, ch, 3, 1, 1)
            ))
            prev_ch = ch
                
    def forward(self, in_x):
        fs = []
        for idx in range(len(self.channels)):
            out_x = getattr(self, f'pyramid{idx+1}')(in_x)
            fs.append(out_x)
            in_x = out_x
        return fs
    
class InitDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(in_ch*2+1, in_ch*2), 
            ResBlock(in_ch*2, skip_ch), 
            nn.ConvTranspose2d(in_ch*2, out_ch+4, 4, 2, 1, bias=True)
        )
    def forward(self, f0, f1, embt):
        h, w = f0.shape[2:]
        embt = embt.repeat(1, 1, h, w)
        out = self.convblock(torch.cat([f0, f1, embt], 1))
        flow0, flow1 = torch.chunk(out[:, :4, ...], 2, 1)
        ft_ = out[:, 4:, ...]
        return flow0, flow1, ft_
    
class IntermediateDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            convrelu(in_ch*3+4, in_ch*3), 
            ResBlock(in_ch*3, skip_ch), 
            nn.ConvTranspose2d(in_ch*3, out_ch+4, 4, 2, 1, bias=True)
        )
    def forward(self, ft_, f0, f1, flow0_in, flow1_in):
        f0_warp = warp(f0, flow0_in)
        f1_warp = warp(f1, flow1_in)
        f_in = torch.cat([ft_, f0_warp, f1_warp, flow0_in, flow1_in], 1)
        out = self.convblock(f_in)
        flow0, flow1 = torch.chunk(out[:, :4, ...], 2, 1)
        ft_ = out[:, 4:, ...]
        flow0 = flow0 + 2.0 * resize(flow0_in, scale_factor=2.0)
        flow1 = flow1 + 2.0 * resize(flow1_in, scale_factor=2.0)
        return flow0, flow1, ft_











def multi_flow_combine(comb_block, img0, img1, flow0, flow1, 
                       mask=None, img_res=None, mean=None):
        '''
            A parallel implementation of multiple flow field warping 
            comb_block: An nn.Seqential object.
            img shape: [b, c, h, w]
            flow shape: [b, 2*num_flows, h, w]
            mask (opt):
                If 'mask' is None, the function conduct a simple average.
            img_res (opt):
                If 'img_res' is None, the function adds zero instead. 
            mean (opt):
                If 'mean' is None, the function adds zero instead.       
        '''
        b, c, h, w = flow0.shape
        num_flows = c // 2
        flow0   =   flow0.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
        flow1   =   flow1.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
        
        mask    =    mask.reshape(b, num_flows, 1, h, w
                            ).reshape(-1, 1, h, w) if mask is not None else None
        img_res = img_res.reshape(b, num_flows, 3, h, w
                            ).reshape(-1, 3, h, w)  if img_res is not None else 0
        img0 = torch.stack([img0] * num_flows, 1).reshape(-1, 3, h, w)
        img1 = torch.stack([img1] * num_flows, 1).reshape(-1, 3, h, w)
        mean = torch.stack([mean] * num_flows, 1).reshape(-1, 1, 1, 1
                                                    ) if mean is not None else 0
        
        img0_warp = warp(img0, flow0)
        img1_warp = warp(img1, flow1)
        img_warps = mask * img0_warp + (1 - mask) * img1_warp + mean + img_res
        img_warps = img_warps.reshape(b, num_flows, 3, h, w)
        imgt_pred = img_warps.mean(1) + comb_block(img_warps.view(b, -1, h, w))
        return imgt_pred


class MultiFlowDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, num_flows=3):
        super(MultiFlowDecoder, self).__init__()
        self.num_flows = num_flows
        self.convblock = nn.Sequential(
            convrelu(in_ch*3+4, in_ch*3), 
            ResBlock(in_ch*3, skip_ch), 
            nn.ConvTranspose2d(in_ch*3, 8*num_flows, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, flow0, flow1):
        n = self.num_flows
        f0_warp = warp(f0, flow0)
        f1_warp = warp(f1, flow1)
        out = self.convblock(torch.cat([ft_, f0_warp, f1_warp, flow0, flow1], 1))
        delta_flow0, delta_flow1, mask, img_res = torch.split(out, [2*n, 2*n, n, 3*n], 1)
        mask = torch.sigmoid(mask)
        
        flow0 = delta_flow0 + 2.0 * resize(flow0, scale_factor=2.0
                                           ).repeat(1, self.num_flows, 1, 1)
        flow1 = delta_flow1 + 2.0 * resize(flow1, scale_factor=2.0
                                           ).repeat(1, self.num_flows, 1, 1)
        
        return flow0, flow1, mask, img_res











def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), 
                            torch.arange(wd, device=device), 
                            indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, cdim, hidden_dim, flow_dim, corr_dim, fc_dim,
                 corr_levels=4, radius=3, scale_factor=None):
        super(SmallUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2 * radius + 1) **2
        self.scale_factor = scale_factor

        self.convc1 = nn.Conv2d(2 * cor_planes, corr_dim, 1, padding=0)
        self.convf1 = nn.Conv2d(4, flow_dim*2, 7, padding=3)
        self.convf2 = nn.Conv2d(flow_dim*2, flow_dim, 3, padding=1)
        self.conv = nn.Conv2d(corr_dim+flow_dim, fc_dim, 3, padding=1)

        self.gru = nn.Sequential(
            nn.Conv2d(fc_dim+4+cdim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
    def forward(self, net, flow, corr):
        net = resize(net, 1 / self.scale_factor
                      ) if self.scale_factor is not None else net
        cor = self.lrelu(self.convc1(corr))
        flo = self.lrelu(self.convf1(flow))
        flo = self.lrelu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow, net], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)
        
        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(delta_flow, scale_factor=self.scale_factor)
        
        return delta_net, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, cdim, hidden_dim, flow_dim, corr_dim, corr_dim2, 
                 fc_dim, corr_levels=4, radius=3, scale_factor=None, out_num=1):
        super(BasicUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2 * radius + 1) **2

        self.scale_factor = scale_factor
        self.convc1 = nn.Conv2d(2 * cor_planes, corr_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(corr_dim, corr_dim2, 3, padding=1)
        self.convf1 = nn.Conv2d(4, flow_dim*2, 7, padding=3)
        self.convf2 = nn.Conv2d(flow_dim*2, flow_dim, 3, padding=1)
        self.conv = nn.Conv2d(flow_dim+corr_dim2, fc_dim, 3, padding=1)

        self.gru = nn.Sequential(
            nn.Conv2d(fc_dim+4+cdim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4*out_num, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
    def forward(self, net, flow, corr):
        net = resize(net, 1 / self.scale_factor
                      ) if self.scale_factor is not None else net
        cor = self.lrelu(self.convc1(corr))
        cor = self.lrelu(self.convc2(cor))
        flo = self.lrelu(self.convf1(flow))
        flo = self.lrelu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow, net], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)
        
        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(delta_flow, scale_factor=self.scale_factor)
        return delta_net, delta_flow


class BidirCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.corr_pyramid_T = []

        corr = BidirCorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr_T = corr.clone().permute(0, 4, 5, 3, 1, 2)

        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        corr_T = corr_T.reshape(batch*h2*w2, dim, h1, w1)
        
        self.corr_pyramid.append(corr)
        self.corr_pyramid_T.append(corr_T)

        for _ in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_T = F.avg_pool2d(corr_T, 2, stride=2)
            self.corr_pyramid.append(corr)
            self.corr_pyramid_T.append(corr_T)

    def __call__(self, coords0, coords1):
        r = self.radius
        coords0 = coords0.permute(0, 2, 3, 1)
        coords1 = coords1.permute(0, 2, 3, 1)
        assert coords0.shape == coords1.shape, f"coords0 shape: [{coords0.shape}] is not equal to [{coords1.shape}]"
        batch, h1, w1, _ = coords0.shape

        out_pyramid = []
        out_pyramid_T = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            corr_T = self.corr_pyramid_T[i]

            dx = torch.linspace(-r, r, 2*r+1, device=coords0.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords0.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

            centroid_lvl_0 = coords0.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            centroid_lvl_1 = coords1.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl_0 = centroid_lvl_0 + delta_lvl
            coords_lvl_1 = centroid_lvl_1 + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl_0)
            corr_T = bilinear_sampler(corr_T, coords_lvl_1)
            corr = corr.view(batch, h1, w1, -1)
            corr_T = corr_T.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
            out_pyramid_T.append(corr_T)

        out = torch.cat(out_pyramid, dim=-1)
        out_T = torch.cat(out_pyramid_T, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float(), out_T.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())











class AMT_S(nn.Module):
    def __init__(self, 
                 corr_radius=3, 
                 corr_lvls=4, 
                 num_flows=3, 
                 channels=[20, 32, 44, 56], 
                 skip_channels=20):
        super(AMT_S, self).__init__()
        self.radius = corr_radius
        self.corr_levels = corr_lvls
        self.num_flows = num_flows
        self.channels = channels
        self.skip_channels = skip_channels

        self.feat_encoder = SmallEncoder(output_dim=84, norm_fn='instance', dropout=0.)
        self.encoder = Encoder(channels)

        self.decoder4 = InitDecoder(channels[3], channels[2], skip_channels)
        self.decoder3 = IntermediateDecoder(channels[2], channels[1], skip_channels)
        self.decoder2 = IntermediateDecoder(channels[1], channels[0], skip_channels)
        self.decoder1 = MultiFlowDecoder(channels[0], skip_channels, num_flows)

        self.update4 = self._get_updateblock(44)
        self.update3 = self._get_updateblock(32, 2)
        self.update2 = self._get_updateblock(20, 4)
        
        self.comb_block = nn.Sequential(
            nn.Conv2d(3*num_flows, 6*num_flows, 3, 1, 1),
            nn.PReLU(6*num_flows),
            nn.Conv2d(6*num_flows, 3, 3, 1, 1),
        )

    def _get_updateblock(self, cdim, scale_factor=None):
        return SmallUpdateBlock(cdim=cdim, hidden_dim=76, flow_dim=20, corr_dim=64, 
                                fc_dim=68, scale_factor=scale_factor, 
                                corr_levels=self.corr_levels, radius=self.radius)

    def _corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        # convert t -> 0 to 0 -> 1 | convert t -> 1 to 1 -> 0
        # based on linear assumption
        t1_scale = 1. / embt
        t0_scale = 1. / (1. - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)
        
        corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale) 
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow

    def forward(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)
        
        fmap0, fmap1 = self.feat_encoder([img0_, img1_]) # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord, 
                                                 up_flow0_4, up_flow1_4, 
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_

        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_ = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_3, up_flow1_3, 
                                                 embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_  = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_2, up_flow1_2, 
                                                 embt, downsample=4)
        
        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_

        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        
        if scale_factor != 1.0: 
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            mask = resize(mask, scale_factor=(1.0/scale_factor))
            img_res = resize(img_res, scale_factor=(1.0/scale_factor))
        
        # Merge multiple predictions 
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1, 
                                                                        mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if eval:
            return  { 'imgt_pred': imgt_pred, }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                'imgt_pred': imgt_pred,
                'flow0_pred': [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                'flow1_pred': [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                'ft_pred': [ft_1_, ft_2_, ft_3_],
            }











class AMT_L(nn.Module):
    def __init__(self, 
                 corr_radius=3, 
                 corr_lvls=4, 
                 num_flows=5,
                 channels=[48, 64, 72, 128], 
                 skip_channels=48
                 ):
        super(AMT_L, self).__init__()
        self.radius = corr_radius
        self.corr_levels = corr_lvls
        self.num_flows = num_flows

        self.feat_encoder = BasicEncoder(output_dim=128, norm_fn='instance', dropout=0.)
        self.encoder = Encoder([48, 64, 72, 128], large=True)
        
        self.decoder4 = InitDecoder(channels[3], channels[2], skip_channels)
        self.decoder3 = IntermediateDecoder(channels[2], channels[1], skip_channels)
        self.decoder2 = IntermediateDecoder(channels[1], channels[0], skip_channels)
        self.decoder1 = MultiFlowDecoder(channels[0], skip_channels, num_flows)

        self.update4 = self._get_updateblock(72, None)
        self.update3 = self._get_updateblock(64, 2.0)
        self.update2 = self._get_updateblock(48, 4.0)
        
        self.comb_block = nn.Sequential(
            nn.Conv2d(3*self.num_flows, 6*self.num_flows, 7, 1, 3),
            nn.PReLU(6*self.num_flows),
            nn.Conv2d(6*self.num_flows, 3, 7, 1, 3),
        )

    def _get_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(cdim=cdim, hidden_dim=128, flow_dim=48, 
                                corr_dim=256, corr_dim2=160, fc_dim=124, 
                                scale_factor=scale_factor, corr_levels=self.corr_levels, 
                                radius=self.radius)

    def _corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        # convert t -> 0 to 0 -> 1 | convert t -> 1 to 1 -> 0
        # based on linear assumption
        t1_scale = 1. / embt
        t0_scale = 1. / (1. - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)
            
        corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale) 
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow
    
    def forward(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)
        
        fmap0, fmap1 = self.feat_encoder([img0_, img1_]) # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord, 
                                                 up_flow0_4, up_flow1_4, 
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_

        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_ = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_3, up_flow1_3, 
                                                 embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_  = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_2, up_flow1_2, 
                                                 embt, downsample=4)
        
        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_

        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        
        if scale_factor != 1.0: 
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            mask = resize(mask, scale_factor=(1.0/scale_factor))
            img_res = resize(img_res, scale_factor=(1.0/scale_factor))

        # Merge multiple predictions 
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1, 
                                                                        mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if eval:
            return  { 'imgt_pred': imgt_pred, }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                'imgt_pred': imgt_pred,
                'flow0_pred': [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                'flow1_pred': [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                'ft_pred': [ft_1_, ft_2_, ft_3_],
            }











class AMT_G(nn.Module):
    def __init__(self, 
                 corr_radius=3, 
                 corr_lvls=4, 
                 num_flows=5, 
                 channels=[84, 96, 112, 128], 
                 skip_channels=84):
        super(AMT_G, self).__init__()
        self.radius = corr_radius
        self.corr_levels = corr_lvls
        self.num_flows = num_flows

        self.feat_encoder = LargeEncoder(output_dim=128, norm_fn='instance', dropout=0.)
        self.encoder = Encoder(channels, large=True)
        self.decoder4 = InitDecoder(channels[3], channels[2], skip_channels)
        self.decoder3 = IntermediateDecoder(channels[2], channels[1], skip_channels)
        self.decoder2 = IntermediateDecoder(channels[1], channels[0], skip_channels)
        self.decoder1 = MultiFlowDecoder(channels[0], skip_channels, num_flows)

        self.update4 = self._get_updateblock(112, None)
        self.update3_low = self._get_updateblock(96, 2.0)
        self.update2_low = self._get_updateblock(84, 4.0)
        
        self.update3_high = self._get_updateblock(96, None)
        self.update2_high = self._get_updateblock(84, None)
        
        self.comb_block = nn.Sequential(
            nn.Conv2d(3*self.num_flows, 6*self.num_flows, 7, 1, 3),
            nn.PReLU(6*self.num_flows),
            nn.Conv2d(6*self.num_flows, 3, 7, 1, 3),
        )

    def _get_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(cdim=cdim, hidden_dim=192, flow_dim=64, 
                                corr_dim=256, corr_dim2=192, fc_dim=188, 
                                scale_factor=scale_factor, corr_levels=self.corr_levels, 
                                radius=self.radius)

    def _corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        # convert t -> 0 to 0 -> 1 | convert t -> 1 to 1 -> 0
        # based on linear assumption
        t1_scale = 1. / embt
        t0_scale = 1. / (1. - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)
            
        corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale) 
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow
    
    def forward(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)
        
        fmap0, fmap1 = self.feat_encoder([img0_, img1_]) # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord, 
                                                 up_flow0_4, up_flow1_4, 
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_

        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_ = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_3, up_flow1_3, 
                                                 embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3_low(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_
        
        # residue update with lookup corr (hr)
        corr_3 = resize(corr_3, scale_factor=2.0)
        up_flow_3 = torch.cat([up_flow0_3, up_flow1_3], dim=1)
        delta_ft_2_, delta_up_flow_3 = self.update3_high(ft_2_, up_flow_3, corr_3)
        ft_2_ += delta_ft_2_
        up_flow0_3 += delta_up_flow_3[:, 0:2]
        up_flow1_3 += delta_up_flow_3[:, 2:4]
        
        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_  = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_2, up_flow1_2, 
                                                 embt, downsample=4)
        
        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2_low(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_
        
        # residue update with lookup corr (hr)
        corr_2 = resize(corr_2, scale_factor=4.0)
        up_flow_2 = torch.cat([up_flow0_2, up_flow1_2], dim=1)
        delta_ft_1_, delta_up_flow_2 = self.update2_high(ft_1_, up_flow_2, corr_2)
        ft_1_ += delta_ft_1_
        up_flow0_2 += delta_up_flow_2[:, 0:2]
        up_flow1_2 += delta_up_flow_2[:, 2:4]
        
        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        
        if scale_factor != 1.0: 
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            mask = resize(mask, scale_factor=(1.0/scale_factor))
            img_res = resize(img_res, scale_factor=(1.0/scale_factor))

        # Merge multiple predictions 
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1, 
                                                                        mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if eval:
            return  { 'imgt_pred': imgt_pred, }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                'imgt_pred': imgt_pred,
                'flow0_pred': [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                'flow1_pred': [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                'ft_pred': [ft_1_, ft_2_, ft_3_],
            }