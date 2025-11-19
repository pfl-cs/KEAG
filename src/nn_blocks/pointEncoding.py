import torch
import torch.nn as nn
import math

class pointEncoding(nn.Module):
	def __init__(self,
				 data_dim,
				 num_maps_each_point,
				 num_samples # d in the paper
				 ):
		super(pointEncoding, self).__init__()
		self.data_dim = data_dim
		self.num_maps_each_point = num_maps_each_point
		self.num_samples = num_samples
		self.omega = nn.Parameter(torch.randn(self.num_samples, requires_grad=True))

	def forward(self, x):
		"""
		:param x: batch_size x num_points, (data_dim * num_maps_each_point)
		:return:
		"""
		# x = torch.repeat_interleave(x, self.num_samples, dim=2)
		# x = x.repeat_interleave(self.num_samples * 2, dim=2)
		x = x.repeat_interleave(self.num_samples, dim=2)

		# print(f'x.shape: {x.shape}')
		# omega_repeat = self.omega.repeat_interleave(2, dim=0).repeat(self.data_dim * 2)
		# # print(f'omega_repeat.shape: {omega_repeat.shape}')
		# encoding = x * omega_repeat
		# encoding[:, :, 0::2] = torch.cos(encoding[:, :, 0::2])
		# encoding[:, :, 1::2] = torch.sin(encoding[:, :, 1::2])

		omega_repeat = self.omega.repeat(self.data_dim * self.num_maps_each_point)
		encoding = x * omega_repeat
		encoding_cos = torch.cos(encoding)
		encoding_sin = torch.sin(encoding)
		encoding = torch.cat([encoding_cos, encoding_sin], dim=2)
		return encoding

