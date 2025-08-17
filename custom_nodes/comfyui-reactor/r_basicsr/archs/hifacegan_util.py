import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# Warning: spectral norm could be buggy
# under eval mode and multi-GPU inference
# A workaround is sticking to single-GPU inference and train mode
from torch.nn.utils import spectral_norm


class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc)
        elif param_free_norm_type == 'syncbatch':
            print('SyncBatchNorm is currently not supported under single-GPU mode, switch to "instance" instead')
            self.param_free_norm = nn.InstanceNorm2d(norm_nc)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(f'{param_free_norm_type} is not a recognized param-free norm type in SPADE')

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128 if norm_nc > 128 else norm_nc

        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * gamma + beta

        return out


class SPADEResnetBlock(nn.Module):
    """
    ResNet block that uses SPADE. It differs from the ResNet block of pix2pixHD in that
    it takes in the segmentation map as input, learns the skip connection if necessary,
    and applies normalization first and then convolution.
    This architecture seemed like a standard architecture for unconditional or
    class-conditional GAN architecture using residual block.
    The code was inspired from https://github.com/LMescheder/GAN_stability.
    """

    def __init__(self, fin, fout, norm_g='spectralspadesyncbatch3x3', semantic_nc=3):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_g:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_g.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.act(self.norm_0(x, seg)))
        dx = self.conv_1(self.act(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def act(self, x):
        return F.leaky_relu(x, 2e-1)


class BaseNetwork(nn.Module):
    """ A basis for hifacegan archs with custom initialization """

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def forward(self, x):
        pass


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x * weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)


class SoftGate(nn.Module):
    COEFF = 12.0

    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)


class SimplifiedLIP(nn.Module):

    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()
        self.logit = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.InstanceNorm2d(channels, affine=True),
            SoftGate())

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac


class LIPEncoder(BaseNetwork):
    """Local Importance-based Pooling (Ziteng Gao et.al.,ICCV 2019)"""

    def __init__(self, input_nc, ngf, sw, sh, n_2xdown, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        # 20200310: Several Convolution (stride 1) + LIP blocks, 4 fold
        kw = 3
        pw = (kw - 1) // 2

        model = [
            nn.Conv2d(input_nc, ngf, kw, stride=1, padding=pw, bias=False),
            norm_layer(ngf),
            nn.ReLU(),
        ]
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio * 2, self.max_ratio)
            model += [
                SimplifiedLIP(ngf * cur_ratio),
                nn.Conv2d(ngf * cur_ratio, ngf * next_ratio, kw, stride=1, padding=pw),
                norm_layer(ngf * next_ratio),
            ]
            cur_ratio = next_ratio
            if i < n_2xdown - 1:
                model += [nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            print('SyncBatchNorm is currently not supported under single-GPU mode, switch to "instance" instead')
            # norm_layer = SynchronizedBatchNorm2d(
            #    get_out_channel(layer), affine=True)
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError(f'normalization layer {subnorm_type} is not recognized')

        return nn.Sequential(layer, norm_layer)

    print('This is a legacy from nvlabs/SPADE, and will be removed in future versions.')
    return add_norm_layer
