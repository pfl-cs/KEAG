import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from nn_blocks import pointEncoding
from utils import ckpt_utils

class simpleScoreNet(nn.Module):
	def __init__(self,
				 data_dim,
				 max_num_candidates,
				 num_config_points_per_candidate,
				 num_maps_each_point,
				 num_geometric_properties,
				 ff_mlp_num_layers,
				 ff_mlp_hidden_dim,
				 use_spatial_points,
				 dropout_rate,
				 use_float64=False):
		super(simpleScoreNet, self).__init__()

		self.data_dim = data_dim
		self.max_num_candidates = max_num_candidates
		self.num_config_points_per_candidate = num_config_points_per_candidate
		self.num_maps_each_point = num_maps_each_point
		self.num_geometric_properties = num_geometric_properties

		self.ff_mlp_num_layers = ff_mlp_num_layers
		self.ff_mlp_hidden_dim = ff_mlp_hidden_dim
		self.use_spatial_points = use_spatial_points

		self.dropout_rate = dropout_rate
		self.use_float64 = use_float64

		if self.use_spatial_points:
			input_dim = (num_config_points_per_candidate * self.data_dim * self.num_maps_each_point
						 + self.num_geometric_properties)
		else:
			input_dim = self.num_geometric_properties
		self.reg = nn.Sequential(
			nn.Linear(input_dim, self.ff_mlp_hidden_dim),
			nn.ReLU(),
			nn.Linear(self.ff_mlp_hidden_dim, self.ff_mlp_hidden_dim),
			nn.ReLU(),
			nn.Linear(self.ff_mlp_hidden_dim, 1)
		)


		if self.use_float64:
			self.double()

	def forward(self, context_rep_points, candi_rep_points, geometric_properties):
		"""
		:param split_context_rep_points: Shape `(batch_size, num_subtree_context_rep_points/num_split_context_rep_points, self.data_dim)
		:param candi_rep_points: Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, self.data_dim)
		:param geometric_properties: Shape `(batch_size, max_num_candidates, self.num_geometric_properties)
		:return:
		"""
		if self.use_spatial_points:
			candi_rep_points = torch.reshape(candi_rep_points,
												   (candi_rep_points.shape[0], self.max_num_candidates, -1))
			states = torch.cat([candi_rep_points, geometric_properties], dim=-1)
		else:
			states = geometric_properties

		scores = self.reg(states)
		scores = scores.squeeze(dim=2)
		return scores
