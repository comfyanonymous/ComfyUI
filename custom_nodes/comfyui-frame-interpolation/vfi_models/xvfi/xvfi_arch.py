import functools, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from comfy.model_management import get_torch_device



class XVFInet(nn.Module):
	
	def __init__(self, args):
		super(XVFInet, self).__init__()
		self.args = args
		self.device = get_torch_device()
		self.nf = args.nf
		self.scale = args.module_scale_factor
		self.vfinet = VFInet(args)
		self.lrelu = nn.ReLU()
		self.in_channels = 3
		self.channel_converter = nn.Sequential(
			nn.Conv3d(self.in_channels, self.nf, [1, 3, 3], [1, 1, 1], [0, 1, 1]),
			nn.ReLU())

		self.rec_ext_ds_module = [self.channel_converter]
		self.rec_ext_ds = nn.Conv3d(self.nf, self.nf, [1, 3, 3], [1, 2, 2], [0, 1, 1])
		for _ in range(int(np.log2(self.scale))):
			self.rec_ext_ds_module.append(self.rec_ext_ds)
			self.rec_ext_ds_module.append(nn.ReLU())
		self.rec_ext_ds_module.append(nn.Conv3d(self.nf, self.nf, [1, 3, 3], 1, [0, 1, 1]))
		self.rec_ext_ds_module.append(RResBlock2D_3D(args, T_reduce_flag=False))
		self.rec_ext_ds_module = nn.Sequential(*self.rec_ext_ds_module)

		self.rec_ctx_ds = nn.Conv3d(self.nf, self.nf, [1, 3, 3], [1, 2, 2], [0, 1, 1])

		print("The lowest scale depth for training (S_trn): ", self.args.S_trn)
		print("The lowest scale depth for test (S_tst): ", self.args.S_tst)

	def forward(self, x, t_value, is_training=True):
		'''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
		B, C, T, H, W = x.size()
		B2, C2 = t_value.size()
		assert C2 == 1, "t_value shape is [B,]"
		assert T % 2 == 0, "T must be an even number"
		t_value = t_value.view(B, 1, 1, 1)

		flow_l = None 
		feat_x = self.rec_ext_ds_module(x)
		feat_x_list = [feat_x]
		self.lowest_depth_level = self.args.S_trn if is_training else self.args.S_tst
		for level in range(1, self.lowest_depth_level+1):
			feat_x = self.rec_ctx_ds(feat_x)
			feat_x_list.append(feat_x)

		if is_training:
			out_l_list = []
			flow_refine_l_list = []
			out_l, flow_l, flow_refine_l = self.vfinet(x, feat_x_list[self.args.S_trn], flow_l, t_value, level=self.args.S_trn, is_training=True)
			out_l_list.append(out_l)
			flow_refine_l_list.append(flow_refine_l)
			for level in range(self.args.S_trn-1, 0, -1): ## self.args.S_trn, self.args.S_trn-1, ..., 1. level 0 is not included
				out_l, flow_l = self.vfinet(x, feat_x_list[level], flow_l, t_value, level=level, is_training=True)
				out_l_list.append(out_l)
			out_l, flow_l, flow_refine_l, occ_0_l0 = self.vfinet(x, feat_x_list[0], flow_l, t_value, level=0, is_training=True)
			out_l_list.append(out_l)
			flow_refine_l_list.append(flow_refine_l)
			return out_l_list[::-1], flow_refine_l_list[::-1], occ_0_l0, torch.mean(x, dim=2) # out_l_list should be reversed. [out_l0, out_l1, ...]

		else: # Testing
			for level in range(self.args.S_tst, 0, -1): ## self.args.S_tst, self.args.S_tst-1, ..., 1. level 0 is not included
				flow_l = self.vfinet(x, feat_x_list[level], flow_l, t_value, level=level, is_training=False)
			out_l = self.vfinet(x, feat_x_list[0], flow_l, t_value, level=0, is_training=False)
			return out_l


class VFInet(nn.Module):
	
	def __init__(self, args):
		super(VFInet, self).__init__()
		self.args = args
		self.device = get_torch_device()
		self.nf = args.nf
		self.scale = args.module_scale_factor
		self.in_channels = 3

		self.conv_flow_bottom = nn.Sequential( 
			nn.Conv2d(2*self.nf, 2*self.nf, [4,4], 2, [1,1]), 
			nn.ReLU(),
			nn.Conv2d(2*self.nf, 4*self.nf, [4,4], 2, [1,1]), 
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 6, [3,3], 1, [1,1]), 
			)

		self.conv_flow1 = nn.Conv2d(2*self.nf, self.nf, [3, 3], 1, [1, 1])
		
		self.conv_flow2 = nn.Sequential(
			nn.Conv2d(2*self.nf + 4, 2 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(2 * self.nf, 4 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 6, [3, 3], 1, [1, 1]),
		)

		self.conv_flow3 = nn.Sequential(
			nn.Conv2d(4 + self.nf * 4, self.nf, [1, 1], 1, [0, 0]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 2 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(2 * self.nf, 4 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
		)
		
		self.refine_unet = RefineUNet(args)
		self.lrelu = nn.ReLU()

	def forward(self, x, feat_x, flow_l_prev, t_value, level, is_training):
		'''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
		B, C, T, H, W = x.size()
		assert T % 2 == 0, "T must be an even number"

		####################### For a single level 
		l = 2 ** level
		x_l = x.permute(0,2,1,3,4)
		x_l = x_l.contiguous().view(B * T, C, H, W)

		if level == 0:
			pass
		else:
			x_l = F.interpolate(x_l, scale_factor=(1.0 / l, 1.0 / l), mode='bicubic', align_corners=False)
		'''
		Down pixel-shuffle
		'''
		x_l = x_l.view(B, T, C, H//l, W//l)
		x_l = x_l.permute(0,2,1,3,4)

		B, C, T, H, W = x_l.size()

		## Feature extraction
		feat0_l = feat_x[:,:,0,:,:]
		feat1_l = feat_x[:,:,1,:,:]

		## Flow estimation
		if flow_l_prev is None:
			flow_l_tmp = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l), dim=1))
			flow_l = flow_l_tmp[:,:4,:,:]
		else:
			up_flow_l_prev = 2.0*F.interpolate(flow_l_prev.detach(), scale_factor=(2,2), mode='bilinear', align_corners=False) 
			warped_feat1_l = self.bwarp(feat1_l, up_flow_l_prev[:,:2,:,:])
			warped_feat0_l = self.bwarp(feat0_l, up_flow_l_prev[:,2:,:,:])
			flow_l_tmp = self.conv_flow2(torch.cat([self.conv_flow1(torch.cat([feat0_l, warped_feat1_l],dim=1)), self.conv_flow1(torch.cat([feat1_l, warped_feat0_l],dim=1)), up_flow_l_prev],dim=1))
			flow_l = flow_l_tmp[:,:4,:,:] + up_flow_l_prev
		
		if not is_training and level!=0: 
			return flow_l 
		
		flow_01_l = flow_l[:,:2,:,:]
		flow_10_l = flow_l[:,2:,:,:]
		z_01_l = torch.sigmoid(flow_l_tmp[:,4:5,:,:])
		z_10_l = torch.sigmoid(flow_l_tmp[:,5:6,:,:])
		
		## Complementary Flow Reversal (CFR)
		flow_forward, norm0_l = self.z_fwarp(flow_01_l, t_value * flow_01_l, z_01_l)  ## Actually, F (t) -> (t+1). Translation only. Not normalized yet
		flow_backward, norm1_l = self.z_fwarp(flow_10_l, (1-t_value) * flow_10_l, z_10_l)  ## Actually, F (1-t) -> (-t). Translation only. Not normalized yet
		
		flow_t0_l = -(1-t_value) * ((t_value)*flow_forward) + (t_value) * ((t_value)*flow_backward) # The numerator of Eq.(1) in the paper.
		flow_t1_l = (1-t_value) * ((1-t_value)*flow_forward) - (t_value) * ((1-t_value)*flow_backward) # The numerator of Eq.(2) in the paper.
		
		norm_l = (1-t_value)*norm0_l + t_value*norm1_l
		mask_ = (norm_l.detach() > 0).type(norm_l.type())
		flow_t0_l = (1-mask_) * flow_t0_l + mask_ * (flow_t0_l.clone() / (norm_l.clone() + (1-mask_))) # Divide the numerator with denominator in Eq.(1)
		flow_t1_l = (1-mask_) * flow_t1_l + mask_ * (flow_t1_l.clone() / (norm_l.clone() + (1-mask_))) # Divide the numerator with denominator in Eq.(2)

		## Feature warping
		warped0_l = self.bwarp(feat0_l, flow_t0_l)
		warped1_l = self.bwarp(feat1_l, flow_t1_l)

		## Flow refinement
		flow_refine_l = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l, flow_t0_l, flow_t1_l], dim=1)
		flow_refine_l = self.conv_flow3(flow_refine_l) + torch.cat([flow_t0_l, flow_t1_l], dim=1)
		flow_t0_l = flow_refine_l[:, :2, :, :]
		flow_t1_l = flow_refine_l[:, 2:4, :, :]

		warped0_l = self.bwarp(feat0_l, flow_t0_l)
		warped1_l = self.bwarp(feat1_l, flow_t1_l)

		## Flow upscale
		flow_t0_l = self.scale * F.interpolate(flow_t0_l, scale_factor=(self.scale, self.scale), mode='bilinear',align_corners=False)
		flow_t1_l = self.scale * F.interpolate(flow_t1_l, scale_factor=(self.scale, self.scale), mode='bilinear',align_corners=False)

		## Image warping and blending
		warped_img0_l = self.bwarp(x_l[:,:,0,:,:], flow_t0_l)
		warped_img1_l = self.bwarp(x_l[:,:,1,:,:], flow_t1_l)
		
		refine_out = self.refine_unet(torch.cat([F.pixel_shuffle(torch.cat([feat0_l, feat1_l, warped0_l, warped1_l],dim=1), self.scale), x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l],dim=1))
		occ_0_l = torch.sigmoid(refine_out[:, 0:1, :, :])
		occ_1_l = 1-occ_0_l
		
		out_l = (1-t_value)*occ_0_l*warped_img0_l + t_value*occ_1_l*warped_img1_l
		out_l = out_l / ( (1-t_value)*occ_0_l + t_value*occ_1_l ) + refine_out[:, 1:4, :, :]

		if not is_training and level==0: 
			return out_l

		if is_training: 
			if flow_l_prev is None:
			# if level == self.args.S_trn:
				return out_l, flow_l, flow_refine_l[:, 0:4, :, :]
			elif level != 0:
				return out_l, flow_l
			else: # level==0
				return out_l, flow_l, flow_refine_l[:, 0:4, :, :], occ_0_l

	def bwarp(self, x, flo):
		'''
		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow
		'''
		B, C, H, W = x.size()
		# mesh grid
		xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
		yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
		grid = torch.cat((xx, yy), 1).float()

		grid = grid.to(self.device)
		vgrid = torch.autograd.Variable(grid) + flo

		# scale grid to [-1,1]
		vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
		vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

		vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
		output = nn.functional.grid_sample(x, vgrid, align_corners=True)
		mask = torch.autograd.Variable(torch.ones(x.size())).to(self.device)
		mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

		# mask[mask<0.9999] = 0
		# mask[mask>0] = 1
		mask = mask.masked_fill_(mask < 0.999, 0)
		mask = mask.masked_fill_(mask > 0, 1)

		return output * mask

	def fwarp(self, img, flo):

		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy
			https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
		"""

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)

		N, C, _, _ = img.size()

		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1:, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)

		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get gaussian weights
		w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

		# secondly, sample each weighted corner
		img11, o11 = self.sample_one(img, x1, y1, w11)
		img12, o12 = self.sample_one(img, x1, y2, w12)
		img21, o21 = self.sample_one(img, x2, y1, w21)
		img22, o22 = self.sample_one(img, x2, y2, w22)

		imgw = img11 + img12 + img21 + img22
		o = o11 + o12 + o21 + o22

		return imgw, o


	def z_fwarp(self, img, flo, z):
		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy
			modified from https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
		"""

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)

		N, C, _, _ = img.size()

		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1:, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)

		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get gaussian weights
		w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2, z+1e-5)

		# secondly, sample each weighted corner
		img11, o11 = self.sample_one(img, x1, y1, w11)
		img12, o12 = self.sample_one(img, x1, y2, w12)
		img21, o21 = self.sample_one(img, x2, y1, w21)
		img22, o22 = self.sample_one(img, x2, y2, w22)

		imgw = img11 + img12 + img21 + img22
		o = o11 + o12 + o21 + o22

		return imgw, o


	def get_gaussian_weights(self, x, y, x1, x2, y1, y2, z=1.0):
		# z 0.0 ~ 1.0
		w11 = z * torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
		w12 = z * torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
		w21 = z * torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
		w22 = z * torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

		return w11, w12, w21, w22

	def sample_one(self, img, shiftx, shifty, weight):
		"""
		Input:
			-img (N, C, H, W)
			-shiftx, shifty (N, c, H, W)
		"""

		N, C, H, W = img.size()

		# flatten all (all restored as Tensors)
		flat_shiftx = shiftx.view(-1)
		flat_shifty = shifty.view(-1)
		flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(self.device).long().repeat(N, C,1,W).view(-1)
		flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(self.device).long().repeat(N, C,H,1).view(-1)
		flat_weight = weight.view(-1)
		flat_img = img.contiguous().view(-1)

		# The corresponding positions in I1
		idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(self.device).long().repeat(1, C, H, W).view(-1)
		idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(self.device).long().repeat(N, 1, H, W).view(-1)
		idxx = flat_shiftx.long() + flat_basex
		idxy = flat_shifty.long() + flat_basey

		# recording the inside part the shifted
		mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

		# Mask off points out of boundaries
		ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
		ids_mask = torch.masked_select(ids, mask).clone().to(self.device)

		# Note here! accmulate fla must be true for proper bp
		img_warp = torch.zeros([N * C * H * W, ]).to(self.device)
		img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

		one_warp = torch.zeros([N * C * H * W, ]).to(self.device)
		one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

		return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)

class RefineUNet(nn.Module):
	def __init__(self, args):
		super(RefineUNet, self).__init__()
		self.args = args
		self.scale = args.module_scale_factor
		self.nf = args.nf
		self.conv1 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.conv2 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.lrelu = nn.ReLU()
		self.NN = nn.UpsamplingNearest2d(scale_factor=2)

		self.enc1 = nn.Conv2d((4*self.nf)//self.scale//self.scale + 4*args.img_ch + 4, self.nf, [4, 4], 2, [1, 1])
		self.enc2 = nn.Conv2d(self.nf, 2*self.nf, [4, 4], 2, [1, 1])
		self.enc3 = nn.Conv2d(2*self.nf, 4*self.nf, [4, 4], 2, [1, 1])
		self.dec0 = nn.Conv2d(4*self.nf, 4*self.nf, [3, 3], 1, [1, 1])
		self.dec1 = nn.Conv2d(4*self.nf + 2*self.nf, 2*self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc2
		self.dec2 = nn.Conv2d(2*self.nf + self.nf, self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc1
		self.dec3 = nn.Conv2d(self.nf, 1+args.img_ch, [3, 3], 1, [1, 1]) ## input added with warped image

	def forward(self, concat):
		enc1 = self.lrelu(self.enc1(concat))
		enc2 = self.lrelu(self.enc2(enc1))
		out = self.lrelu(self.enc3(enc2))

		out = self.lrelu(self.dec0(out))
		out = self.NN(out)

		out = torch.cat((out,enc2),dim=1)
		out = self.lrelu(self.dec1(out))

		out = self.NN(out)
		out = torch.cat((out,enc1),dim=1)
		out = self.lrelu(self.dec2(out))

		out = self.NN(out)
		out = self.dec3(out)
		return out

class ResBlock2D_3D(nn.Module):
	## Shape of input [B,C,T,H,W]
	## Shape of output [B,C,T,H,W]
	def __init__(self, args):
		super(ResBlock2D_3D, self).__init__()
		self.args = args
		self.nf = args.nf

		self.conv3x3_1 = nn.Conv3d(self.nf, self.nf, [1,3,3], 1, [0,1,1])
		self.conv3x3_2 = nn.Conv3d(self.nf, self.nf, [1,3,3], 1, [0,1,1])
		self.lrelu = nn.ReLU()

	def forward(self, x):
		'''
		x shape : [B,C,T,H,W]
		'''
		B, C, T, H, W = x.size()

		out = self.conv3x3_2(self.lrelu(self.conv3x3_1(x)))

		return x + out

class RResBlock2D_3D(nn.Module):
	
	def __init__(self, args, T_reduce_flag=False):
		super(RResBlock2D_3D, self).__init__()
		self.args = args
		self.nf = args.nf
		self.T_reduce_flag = T_reduce_flag
		self.resblock1 = ResBlock2D_3D(self.args)
		self.resblock2 = ResBlock2D_3D(self.args)
		if T_reduce_flag:
			self.reduceT_conv = nn.Conv3d(self.nf, self.nf, [3,1,1], 1, [0,0,0])

	def forward(self, x):
		'''
		x shape : [B,C,T,H,W]
		'''
		out = self.resblock1(x)
		out = self.resblock2(out)
		if self.T_reduce_flag:
			return self.reduceT_conv(out + x)
		else:
			return out + x

def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('Conv3d') != -1):
        init.xavier_normal_(m.weight)
        # init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)