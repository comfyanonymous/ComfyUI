"""
https://github.com/dajes/frame-interpolation-pytorch/blob/main/feature_extractor.py
https://github.com/dajes/frame-interpolation-pytorch/blob/main/fusion.py
https://github.com/dajes/frame-interpolation-pytorch/blob/main/interpolator.py
https://github.com/dajes/frame-interpolation-pytorch/blob/main/pyramid_flow_estimator.py
https://github.com/dajes/frame-interpolation-pytorch/blob/main/util.py
"""

"""PyTorch layer for extracting image features for the film_net interpolator.

The feature extractor implemented here converts an image pyramid into a pyramid
of deep features. The feature pyramid serves a similar purpose as U-Net
architecture's encoder, but we use a special cascaded architecture described in
Multi-view Image Fusion [1].

For comprehensiveness, below is a short description of the idea. While the
description is a bit involved, the cascaded feature pyramid can be used just
like any image feature pyramid.

Why cascaded architeture?
=========================
To understand the concept it is worth reviewing a traditional feature pyramid
first: *A traditional feature pyramid* as in U-net or in many optical flow
networks is built by alternating between convolutions and pooling, starting
from the input image.

It is well known that early features of such architecture correspond to low
level concepts such as edges in the image whereas later layers extract
semantically higher level concepts such as object classes etc. In other words,
the meaning of the filters in each resolution level is different. For problems
such as semantic segmentation and many others this is a desirable property.

However, the asymmetric features preclude sharing weights across resolution
levels in the feature extractor itself and in any subsequent neural networks
that follow. This can be a downside, since optical flow prediction, for
instance is symmetric across resolution levels. The cascaded feature
architecture addresses this shortcoming.

How is it built?
================
The *cascaded* feature pyramid contains feature vectors that have constant
length and meaning on each resolution level, except few of the finest ones. The
advantage of this is that the subsequent optical flow layer can learn
synergically from many resolutions. This means that coarse level prediction can
benefit from finer resolution training examples, which can be useful with
moderately sized datasets to avoid overfitting.

The cascaded feature pyramid is built by extracting shallower subtree pyramids,
each one of them similar to the traditional architecture. Each subtree
pyramid S_i is extracted starting from each resolution level:

image resolution 0 -> S_0
image resolution 1 -> S_1
image resolution 2 -> S_2
...

If we denote the features at level j of subtree i as S_i_j, the cascaded pyramid
is constructed by concatenating features as follows (assuming subtree depth=3):

lvl
feat_0 = concat(                               S_0_0 )
feat_1 = concat(                         S_1_0 S_0_1 )
feat_2 = concat(                   S_2_0 S_1_1 S_0_2 )
feat_3 = concat(             S_3_0 S_2_1 S_1_2       )
feat_4 = concat(       S_4_0 S_3_1 S_2_2             )
feat_5 = concat( S_5_0 S_4_1 S_3_2                   )
   ....

In above, all levels except feat_0 and feat_1 have the same number of features
with similar semantic meaning. This enables training a single optical flow
predictor module shared by levels 2,3,4,5... . For more details and evaluation
see [1].

[1] Multi-view Image Fusion, Trinidad et al. 2019
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class SubTreeExtractor(nn.Module):
    """Extracts a hierarchical set of features from an image.

    This is a conventional, hierarchical image feature extractor, that extracts
    [k, k*2, k*4... ] filters for the image pyramid where k=options.sub_levels.
    Each level is followed by average pooling.
    """

    def __init__(self, in_channels=3, channels=64, n_layers=4):
        super().__init__()
        convs = []
        for i in range(n_layers):
            convs.append(nn.Sequential(
                conv(in_channels, (channels << i), 3),
                conv((channels << i), (channels << i), 3)
            ))
            in_channels = channels << i
        self.convs = nn.ModuleList(convs)

    def forward(self, image: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Extracts a pyramid of features from the image.

        Args:
          image: TORCH.Tensor with shape BATCH_SIZE x HEIGHT x WIDTH x CHANNELS.
          n: number of pyramid levels to extract. This can be less or equal to
           options.sub_levels given in the __init__.
        Returns:
          The pyramid of features, starting from the finest level. Each element
          contains the output after the last convolution on the corresponding
          pyramid level.
        """
        head = image
        pyramid = []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, kernel_size=2, stride=2)
        return pyramid


class FeatureExtractor(nn.Module):
    """Extracts features from an image pyramid using a cascaded architecture.
    """

    def __init__(self, in_channels=3, channels=64, sub_levels=4):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extracts a cascaded feature pyramid.

        Args:
          image_pyramid: Image pyramid as a list, starting from the finest level.
        Returns:
          A pyramid of cascaded features.
        """
        sub_pyramids: List[List[torch.Tensor]] = []
        for i in range(len(image_pyramid)):
            # At each level of the image pyramid, creates a sub_pyramid of features
            # with 'sub_levels' pyramid levels, re-using the same SubTreeExtractor.
            # We use the same instance since we want to share the weights.
            #
            # However, we cap the depth of the sub_pyramid so we don't create features
            # that are beyond the coarsest level of the cascaded feature pyramid we
            # want to generate.
            capped_sub_levels = min(len(image_pyramid) - i, self.sub_levels)
            sub_pyramids.append(self.extract_sublevels(image_pyramid[i], capped_sub_levels))
        # Below we generate the cascades of features on each level of the feature
        # pyramid. Assuming sub_levels=3, The layout of the features will be
        # as shown in the example on file documentation above.
        feature_pyramid: List[torch.Tensor] = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid











"""The final fusion stage for the film_net frame interpolator.

The inputs to this module are the warped input images, image features and
flow fields, all aligned to the target frame (often midway point between the
two original inputs). The output is the final image. FILM has no explicit
occlusion handling -- instead using the abovementioned information this module
automatically decides how to best blend the inputs together to produce content
in areas where the pixels can only be borrowed from one of the inputs.

Similarly, this module also decides on how much to blend in each input in case
of fractional timestep that is not at the halfway point. For example, if the two
inputs images are at t=0 and t=1, and we were to synthesize a frame at t=0.1,
it often makes most sense to favor the first input. However, this is not
always the case -- in particular in occluded pixels.

The architecture of the Fusion module follows U-net [1] architecture's decoder
side, e.g. each pyramid level consists of concatenation with upsampled coarser
level output, and two 3x3 convolutions.

The upsampling is implemented as 'resize convolution', e.g. nearest neighbor
upsampling followed by 2x2 convolution as explained in [2]. The classic U-net
uses max-pooling which has a tendency to create checkerboard artifacts.

[1] Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
    Segmentation, 2015, https://arxiv.org/pdf/1505.04597.pdf
[2] https://distill.pub/2016/deconv-checkerboard/
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


_NUMBER_OF_COLOR_CHANNELS = 3


def get_channels_at_level(level, filters):
    n_images = 2
    channels = _NUMBER_OF_COLOR_CHANNELS
    flows = 2

    return (sum(filters << i for i in range(level)) + channels + flows) * n_images


class Fusion(nn.Module):
    """The decoder."""

    def __init__(self, n_layers=4, specialized_layers=3, filters=64):
        """
        Args:
            m: specialized levels
        """
        super().__init__()

        # The final convolution that outputs RGB:
        self.output_conv = nn.Conv2d(filters, 3, kernel_size=1)

        # Each item 'convs[i]' will contain the list of convolutions to be applied
        # for pyramid level 'i'.
        self.convs = nn.ModuleList()

        # Create the convolutions. Roughly following the feature extractor, we
        # double the number of filters when the resolution halves, but only up to
        # the specialized_levels, after which we use the same number of filters on
        # all levels.
        #
        # We create the convs in fine-to-coarse order, so that the array index
        # for the convs will correspond to our normal indexing (0=finest level).
        # in_channels: tuple = (128, 202, 256, 522, 512, 1162, 1930, 2442)

        in_channels = get_channels_at_level(n_layers, filters)
        increase = 0
        for i in range(n_layers)[::-1]:
            num_filters = (filters << i) if i < specialized_layers else (filters << specialized_layers)
            convs = nn.ModuleList([
                conv(in_channels, num_filters, size=2, activation=None),
                conv(in_channels + (increase or num_filters), num_filters, size=3),
                conv(num_filters, num_filters, size=3)]
            )
            self.convs.append(convs)
            in_channels = num_filters
            increase = get_channels_at_level(i, filters) - num_filters // 2

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Runs the fusion module.

        Args:
          pyramid: The input feature pyramid as list of tensors. Each tensor being
            in (B x H x W x C) format, with finest level tensor first.

        Returns:
          A batch of RGB images.
        Raises:
          ValueError, if len(pyramid) != config.fusion_pyramid_levels as provided in
            the constructor.
        """

        # As a slight difference to a conventional decoder (e.g. U-net), we don't
        # apply any extra convolutions to the coarsest level, but just pass it
        # to finer levels for concatenation. This choice has not been thoroughly
        # evaluated, but is motivated by the educated guess that the fusion part
        # probably does not need large spatial context, because at this point the
        # features are spatially aligned by the preceding warp.
        net = pyramid[-1]

        # Loop starting from the 2nd coarsest level:
        # for i in reversed(range(0, len(pyramid) - 1)):
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            # Resize the tensor from coarser level to match for concatenation.
            level_size = pyramid[i].shape[2:4]
            net = F.interpolate(net, size=level_size, mode='nearest')
            net = layers[0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = layers[1](net)
            net = layers[2](net)
        net = self.output_conv(net)
        return net











"""The film_net frame interpolator main model code.

Basics
======
The film_net is an end-to-end learned neural frame interpolator implemented as
a PyTorch model. It has the following inputs and outputs:

Inputs:
  x0: image A.
  x1: image B.
  time: desired sub-frame time.

Outputs:
  image: the predicted in-between image at the chosen time in range [0, 1].

Additional outputs include forward and backward warped image pyramids, flow
pyramids, etc., that can be visualized for debugging and analysis.

Note that many training sets only contain triplets with ground truth at
time=0.5. If a model has been trained with such training set, it will only work
well for synthesizing frames at time=0.5. Such models can only generate more
in-between frames using recursion.

Architecture
============
The inference consists of three main stages: 1) feature extraction 2) warping
3) fusion. On high-level, the architecture has similarities to Context-aware
Synthesis for Video Frame Interpolation [1], but the exact architecture is
closer to Multi-view Image Fusion [2] with some modifications for the frame
interpolation use-case.

Feature extraction stage employs the cascaded multi-scale architecture described
in [2]. The advantage of this architecture is that coarse level flow prediction
can be learned from finer resolution image samples. This is especially useful
to avoid overfitting with moderately sized datasets.

The warping stage uses a residual flow prediction idea that is similar to
PWC-Net [3], Multi-view Image Fusion [2] and many others.

The fusion stage is similar to U-Net's decoder where the skip connections are
connected to warped image and feature pyramids. This is described in [2].

Implementation Conventions
====================
Pyramids
--------
Throughtout the model, all image and feature pyramids are stored as python lists
with finest level first followed by downscaled versions obtained by successively
halving the resolution. The depths of all pyramids are determined by
options.pyramid_levels. The only exception to this is internal to the feature
extractor, where smaller feature pyramids are temporarily constructed with depth
options.sub_levels.

Color ranges & gamma
--------------------
The model code makes no assumptions on whether the images are in gamma or
linearized space or what is the range of RGB color values. So a model can be
trained with different choices. This does not mean that all the choices lead to
similar results. In practice the model has been proven to work well with RGB
scale = [0,1] with gamma-space images (i.e. not linearized).

[1] Context-aware Synthesis for Video Frame Interpolation, Niklaus and Liu, 2018
[2] Multi-view Image Fusion, Trinidad et al, 2019
[3] PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
"""
from typing import Dict, List

import torch
from torch import nn



class Interpolator(nn.Module):
    def __init__(
            self,
            pyramid_levels=7,
            fusion_pyramid_levels=5,
            specialized_levels=3,
            sub_levels=4,
            filters=64,
            flow_convs=(3, 3, 3, 3),
            flow_filters=(32, 64, 128, 256),
    ):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels

        self.extract = FeatureExtractor(3, filters, sub_levels)
        self.predict_flow = PyramidFlowEstimator(filters, flow_convs, flow_filters)
        self.fuse = Fusion(sub_levels, specialized_levels, filters)

    def shuffle_images(self, x0, x1):
        return [
            build_image_pyramid(x0, self.pyramid_levels),
            build_image_pyramid(x1, self.pyramid_levels)
        ]

    def debug_forward(self, x0, x1, batch_dt) -> Dict[str, List[torch.Tensor]]:
        image_pyramids = self.shuffle_images(x0, x1)

        # Siamese feature pyramids:
        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]

        # Predict forward flow.
        forward_residual_flow_pyramid = self.predict_flow(feature_pyramids[0], feature_pyramids[1])

        # Predict backward flow.
        backward_residual_flow_pyramid = self.predict_flow(feature_pyramids[1], feature_pyramids[0])

        # Concatenate features and images:

        # Note that we keep up to 'fusion_pyramid_levels' levels as only those
        # are used by the fusion module.

        forward_flow_pyramid = flow_pyramid_synthesis(forward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        backward_flow_pyramid = flow_pyramid_synthesis(backward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        # We multiply the flows with t and 1-t to warp to the desired fractional time.
        #
        # Note: In film_net we fix time to be 0.5, and recursively invoke the interpo-
        # lator for multi-frame interpolation. Below, we create a constant tensor of
        # shape [B]. We use the `time` tensor to infer the batch size.
        mid_time = torch.full_like(batch_dt, .5)
        backward_flow = multiply_pyramid(backward_flow_pyramid, mid_time[:, 0])
        forward_flow = multiply_pyramid(forward_flow_pyramid, 1 - mid_time[:, 0])

        pyramids_to_warp = [
            concatenate_pyramids(image_pyramids[0][:self.fusion_pyramid_levels],
                                      feature_pyramids[0][:self.fusion_pyramid_levels]),
            concatenate_pyramids(image_pyramids[1][:self.fusion_pyramid_levels],
                                      feature_pyramids[1][:self.fusion_pyramid_levels])
        ]

        # Warp features and images using the flow. Note that we use backward warping
        # and backward flow is used to read from image 0 and forward flow from
        # image 1.
        forward_warped_pyramid = pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid = pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned_pyramid = concatenate_pyramids(forward_warped_pyramid,
                                                    backward_warped_pyramid)
        aligned_pyramid = concatenate_pyramids(aligned_pyramid, backward_flow)
        aligned_pyramid = concatenate_pyramids(aligned_pyramid, forward_flow)

        return {
            'image': [self.fuse(aligned_pyramid)],
            'forward_residual_flow_pyramid': forward_residual_flow_pyramid,
            'backward_residual_flow_pyramid': backward_residual_flow_pyramid,
            'forward_flow_pyramid': forward_flow_pyramid,
            'backward_flow_pyramid': backward_flow_pyramid,
        }


    def forward(self, x0, x1, batch_dt) -> torch.Tensor:
        return self.debug_forward(x0, x1, batch_dt)['image'][0]










"""PyTorch layer for estimating optical flow by a residual flow pyramid.

This approach of estimating optical flow between two images can be traced back
to [1], but is also used by later neural optical flow computation methods such
as SpyNet [2] and PWC-Net [3].

The basic idea is that the optical flow is first estimated in a coarse
resolution, then the flow is upsampled to warp the higher resolution image and
then a residual correction is computed and added to the estimated flow. This
process is repeated in a pyramid on coarse to fine order to successively
increase the resolution of both optical flow and the warped image.

In here, the optical flow predictor is used as an internal component for the
film_net frame interpolator, to warp the two input images into the inbetween,
target frame.

[1] F. Glazer, Hierarchical motion detection. PhD thesis, 1987.
[2] A. Ranjan and M. J. Black, Optical Flow Estimation using a Spatial Pyramid
    Network. 2016
[3] D. Sun X. Yang, M-Y. Liu and J. Kautz, PWC-Net: CNNs for Optical Flow Using
    Pyramid, Warping, and Cost Volume, 2017
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F



class FlowEstimator(nn.Module):
    """Small-receptive field predictor for computing the flow between two images.

    This is used to compute the residual flow fields in PyramidFlowEstimator.

    Note that while the number of 3x3 convolutions & filters to apply is
    configurable, two extra 1x1 convolutions are appended to extract the flow in
    the end.

    Attributes:
      name: The name of the layer
      num_convs: Number of 3x3 convolutions to apply
      num_filters: Number of filters in each 3x3 convolution
    """

    def __init__(self, in_channels: int, num_convs: int, num_filters: int):
        super(FlowEstimator, self).__init__()

        self._convs = nn.ModuleList()
        for i in range(num_convs):
            self._convs.append(conv(in_channels=in_channels, out_channels=num_filters, size=3))
            in_channels = num_filters
        self._convs.append(conv(in_channels, num_filters // 2, size=1))
        in_channels = num_filters // 2
        # For the final convolution, we want no activation at all to predict the
        # optical flow vector values. We have done extensive testing on explicitly
        # bounding these values using sigmoid, but it turned out that having no
        # activation gives better results.
        self._convs.append(conv(in_channels, 2, size=1, activation=None))

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Estimates optical flow between two images.

        Args:
          features_a: per pixel feature vectors for image A (B x H x W x C)
          features_b: per pixel feature vectors for image B (B x H x W x C)

        Returns:
          A tensor with optical flow from A to B
        """
        net = torch.cat([features_a, features_b], dim=1)
        for conv in self._convs:
            net = conv(net)
        return net


class PyramidFlowEstimator(nn.Module):
    """Predicts optical flow by coarse-to-fine refinement.
    """

    def __init__(self, filters: int = 64,
                 flow_convs: tuple = (3, 3, 3, 3),
                 flow_filters: tuple = (32, 64, 128, 256)):
        super(PyramidFlowEstimator, self).__init__()

        in_channels = filters << 1
        predictors = []
        for i in range(len(flow_convs)):
            predictors.append(
                FlowEstimator(
                    in_channels=in_channels,
                    num_convs=flow_convs[i],
                    num_filters=flow_filters[i]))
            in_channels += filters << (i + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a: List[torch.Tensor],
                feature_pyramid_b: List[torch.Tensor]) -> List[torch.Tensor]:
        """Estimates residual flow pyramids between two image pyramids.

        Each image pyramid is represented as a list of tensors in fine-to-coarse
        order. Each individual image is represented as a tensor where each pixel is
        a vector of image features.

        flow_pyramid_synthesis can be used to convert the residual flow
        pyramid returned by this method into a flow pyramid, where each level
        encodes the flow instead of a residual correction.

        Args:
          feature_pyramid_a: image pyramid as a list in fine-to-coarse order
          feature_pyramid_b: image pyramid as a list in fine-to-coarse order

        Returns:
          List of flow tensors, in fine-to-coarse order, each level encoding the
          difference against the bilinearly upsampled version from the coarser
          level. The coarsest flow tensor, e.g. the last element in the array is the
          'DC-term', e.g. not a residual (alternatively you can think of it being a
          residual against zero).
        """
        levels = len(feature_pyramid_a)
        v = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in range(levels - 2, len(self._predictors) - 1, -1):
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = self._predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v

        for k, predictor in enumerate(self._predictors):
            i = len(self._predictors) - 1 - k
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v
        return residuals










"""Various utilities used in the film_net frame interpolator model."""
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region


def load_image(path, align=64):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region


def build_image_pyramid(image: torch.Tensor, pyramid_levels: int = 3) -> List[torch.Tensor]:
    """Builds an image pyramid from a given image.

    The original image is included in the pyramid and the rest are generated by
    successively halving the resolution.

    Args:
      image: the input image.
      options: film_net options object

    Returns:
      A list of images starting from the finest with options.pyramid_levels items
    """

    pyramid = []
    for i in range(pyramid_levels):
        pyramid.append(image)
        if i < pyramid_levels - 1:
            image = F.avg_pool2d(image, 2, 2)
    return pyramid


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warps the image using the given flow.

    Specifically, the output pixel in batch b, at position x, y will be computed
    as follows:
      (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
      output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

    Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
    y in position 1.

    Args:
      image: An image with shape BxHxWxC.
      flow: A flow with shape BxHxWx2, with the two channels denoting the relative
        offset in order: (dx, dy).
    Returns:
      A warped image.
    """
    flow = -flow.flip(1)

    dtype = flow.dtype
    device = flow.device

    # warped = tfa_image.dense_image_warp(image, flow)
    # Same as above but with pytorch
    ls1 = 1 - 1 / flow.shape[3]
    ls2 = 1 - 1 / flow.shape[2]

    normalized_flow2 = flow.permute(0, 2, 3, 1) / torch.tensor(
        [flow.shape[2] * .5, flow.shape[3] * .5], dtype=dtype, device=device)[None, None, None]
    normalized_flow2 = torch.stack([
        torch.linspace(-ls1, ls1, flow.shape[3], dtype=dtype, device=device)[None, None, :] - normalized_flow2[..., 1],
        torch.linspace(-ls2, ls2, flow.shape[2], dtype=dtype, device=device)[None, :, None] - normalized_flow2[..., 0],
    ], dim=3)

    padding_mode = "border"
    if device.type == "mps":
        # https://github.com/pytorch/pytorch/issues/125098
        padding_mode = "zeros"
        normalized_flow2 = normalized_flow2.clamp(-1, 1)
    warped = F.grid_sample(
        input=image,
        grid=normalized_flow2,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=False,
    )
    return warped.reshape(image.shape)


def multiply_pyramid(pyramid: List[torch.Tensor],
                     scalar: torch.Tensor) -> List[torch.Tensor]:
    """Multiplies all image batches in the pyramid by a batch of scalars.

    Args:
      pyramid: Pyramid of image batches.
      scalar: Batch of scalars.

    Returns:
      An image pyramid with all images multiplied by the scalar.
    """
    # To multiply each image with its corresponding scalar, we first transpose
    # the batch of images from BxHxWxC-format to CxHxWxB. This can then be
    # multiplied with a batch of scalars, then we transpose back to the standard
    # BxHxWxC form.
    return [image * scalar for image in pyramid]


def flow_pyramid_synthesis(
        residual_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Converts a residual flow pyramid into a flow pyramid."""
    flow = residual_pyramid[-1]
    flow_pyramid: List[torch.Tensor] = [flow]
    for residual_flow in residual_pyramid[:-1][::-1]:
        level_size = residual_flow.shape[2:4]
        flow = F.interpolate(2 * flow, size=level_size, mode='bilinear')
        flow = residual_flow + flow
        flow_pyramid.insert(0, flow)
    return flow_pyramid


def pyramid_warp(feature_pyramid: List[torch.Tensor],
                 flow_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Warps the feature pyramid using the flow pyramid.

    Args:
      feature_pyramid: feature pyramid starting from the finest level.
      flow_pyramid: flow fields, starting from the finest level.

    Returns:
      Reverse warped feature pyramid.
    """
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(features, flow))
    return warped_feature_pyramid


def concatenate_pyramids(pyramid1: List[torch.Tensor],
                         pyramid2: List[torch.Tensor]) -> List[torch.Tensor]:
    """Concatenates each pyramid level together in the channel dimension."""
    result = []
    for features1, features2 in zip(pyramid1, pyramid2):
        result.append(torch.cat([features1, features2], dim=1))
    return result


def conv(in_channels, out_channels, size, activation: Optional[str] = 'relu'):
    # Since PyTorch doesn't have an in-built activation in Conv2d, we use a
    # Sequential layer to combine Conv2d and Leaky ReLU in one module.
    _conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=size,
        padding='same')
    if activation is None:
        return _conv
    assert activation == 'relu'
    return nn.Sequential(
        _conv,
        nn.LeakyReLU(.2)
    )