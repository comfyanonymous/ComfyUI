import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm

from r_basicsr.utils.registry import ARCH_REGISTRY
from .dfdnet_util import AttentionBlock, Blur, MSDilationBlock, UpResBlock, adaptive_instance_normalization
from .vgg_arch import VGGFeatureExtractor


class SFTUpBlock(nn.Module):
    """Spatial feature transform (SFT) with upsampling block.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size in convolutions. Default: 3.
        padding (int): Padding in convolutions. Default: 1.
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(SFTUpBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Blur(in_channel),
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.04, True),
            # The official codes use two LeakyReLU here, so 0.04 for equivalent
        )
        self.convup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2, True),
        )

        # for SFT scale and shift
        self.scale_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)))
        self.shift_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.Sigmoid())
        # The official codes use sigmoid for shift block, do not know why

    def forward(self, x, updated_feat):
        out = self.conv1(x)
        # SFT
        scale = self.scale_block(updated_feat)
        shift = self.shift_block(updated_feat)
        out = out * scale + shift
        # upsample
        out = self.convup(out)
        return out


@ARCH_REGISTRY.register()
class DFDNet(nn.Module):
    """DFDNet: Deep Face Dictionary Network.

    It only processes faces with 512x512 size.

    Args:
        num_feat (int): Number of feature channels.
        dict_path (str): Path to the facial component dictionary.
    """

    def __init__(self, num_feat, dict_path):
        super().__init__()
        self.parts = ['left_eye', 'right_eye', 'nose', 'mouth']
        # part_sizes: [80, 80, 50, 110]
        channel_sizes = [128, 256, 512, 512]
        self.feature_sizes = np.array([256, 128, 64, 32])
        self.vgg_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'conv5_4']
        self.flag_dict_device = False

        # dict
        self.dict = torch.load(dict_path)

        # vgg face extractor
        self.vgg_extractor = VGGFeatureExtractor(
            layer_name_list=self.vgg_layers,
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=True,
            requires_grad=False)

        # attention block for fusing dictionary features and input features
        self.attn_blocks = nn.ModuleDict()
        for idx, feat_size in enumerate(self.feature_sizes):
            for name in self.parts:
                self.attn_blocks[f'{name}_{feat_size}'] = AttentionBlock(channel_sizes[idx])

        # multi scale dilation block
        self.multi_scale_dilation = MSDilationBlock(num_feat * 8, dilation=[4, 3, 2, 1])

        # upsampling and reconstruction
        self.upsample0 = SFTUpBlock(num_feat * 8, num_feat * 8)
        self.upsample1 = SFTUpBlock(num_feat * 8, num_feat * 4)
        self.upsample2 = SFTUpBlock(num_feat * 4, num_feat * 2)
        self.upsample3 = SFTUpBlock(num_feat * 2, num_feat)
        self.upsample4 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1)), nn.LeakyReLU(0.2, True), UpResBlock(num_feat),
            UpResBlock(num_feat), nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def swap_feat(self, vgg_feat, updated_feat, dict_feat, location, part_name, f_size):
        """swap the features from the dictionary."""
        # get the original vgg features
        part_feat = vgg_feat[:, :, location[1]:location[3], location[0]:location[2]].clone()
        # resize original vgg features
        part_resize_feat = F.interpolate(part_feat, dict_feat.size()[2:4], mode='bilinear', align_corners=False)
        # use adaptive instance normalization to adjust color and illuminations
        dict_feat = adaptive_instance_normalization(dict_feat, part_resize_feat)
        # get similarity scores
        similarity_score = F.conv2d(part_resize_feat, dict_feat)
        similarity_score = F.softmax(similarity_score.view(-1), dim=0)
        # select the most similar features in the dict (after norm)
        select_idx = torch.argmax(similarity_score)
        swap_feat = F.interpolate(dict_feat[select_idx:select_idx + 1], part_feat.size()[2:4])
        # attention
        attn = self.attn_blocks[f'{part_name}_' + str(f_size)](swap_feat - part_feat)
        attn_feat = attn * swap_feat
        # update features
        updated_feat[:, :, location[1]:location[3], location[0]:location[2]] = attn_feat + part_feat
        return updated_feat

    def put_dict_to_device(self, x):
        if self.flag_dict_device is False:
            for k, v in self.dict.items():
                for kk, vv in v.items():
                    self.dict[k][kk] = vv.to(x)
            self.flag_dict_device = True

    def forward(self, x, part_locations):
        """
        Now only support testing with batch size = 0.

        Args:
            x (Tensor): Input faces with shape (b, c, 512, 512).
            part_locations (list[Tensor]): Part locations.
        """
        self.put_dict_to_device(x)
        # extract vggface features
        vgg_features = self.vgg_extractor(x)
        # update vggface features using the dictionary for each part
        updated_vgg_features = []
        batch = 0  # only supports testing with batch size = 0
        for vgg_layer, f_size in zip(self.vgg_layers, self.feature_sizes):
            dict_features = self.dict[f'{f_size}']
            vgg_feat = vgg_features[vgg_layer]
            updated_feat = vgg_feat.clone()

            # swap features from dictionary
            for part_idx, part_name in enumerate(self.parts):
                location = (part_locations[part_idx][batch] // (512 / f_size)).int()
                updated_feat = self.swap_feat(vgg_feat, updated_feat, dict_features[part_name], location, part_name,
                                              f_size)

            updated_vgg_features.append(updated_feat)

        vgg_feat_dilation = self.multi_scale_dilation(vgg_features['conv5_4'])
        # use updated vgg features to modulate the upsampled features with
        # SFT (Spatial Feature Transform) scaling and shifting manner.
        upsampled_feat = self.upsample0(vgg_feat_dilation, updated_vgg_features[3])
        upsampled_feat = self.upsample1(upsampled_feat, updated_vgg_features[2])
        upsampled_feat = self.upsample2(upsampled_feat, updated_vgg_features[1])
        upsampled_feat = self.upsample3(upsampled_feat, updated_vgg_features[0])
        out = self.upsample4(upsampled_feat)

        return out
