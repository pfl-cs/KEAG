import torch
import torch.nn as nn
import torch.utils.data
from nn_blocks import pointEncoding
from nn_blocks import attn

class KEAG(nn.Module):
	def __init__(self,
				 task,
				 data_dim,
				 max_num_candidates,
				 num_keep_attn_weights,
				 num_config_points_per_candidate,
				 num_maps_each_point,
				 num_geometric_properties,
				 num_point_encoding_samples,
				 num_attn_layers,
				 num_attn_heads,
				 attn_mlp_hidden_dim,
				 attn_mlp_output_dim,
				 ff_mlp_num_layers,
				 ff_mlp_hidden_dim,
				 dropout_rate,
				 use_attn_weights_as_feature=False,
				 use_float64=False):
		super(KEAG, self).__init__()

		self.task = task
		self.data_dim = data_dim
		self.max_num_candidates = max_num_candidates
		self.num_keep_attn_weights = num_keep_attn_weights
		self.num_config_points_per_candidate = num_config_points_per_candidate
		self.num_maps_each_point = num_maps_each_point
		self.num_geometric_properties = num_geometric_properties
		self.num_point_encoding_samples = num_point_encoding_samples

		self.num_attn_layers = num_attn_layers
		self.num_attn_heads = num_attn_heads

		self.attn_head_key_dim = data_dim * num_maps_each_point * num_point_encoding_samples * 2
		while attn_mlp_hidden_dim < self.attn_head_key_dim:
			attn_mlp_hidden_dim *= 2
		self.attn_mlp_hidden_dim = attn_mlp_hidden_dim
		self.attn_mlp_output_dim = attn_mlp_output_dim

		self.ff_mlp_num_layers = ff_mlp_num_layers
		self.ff_mlp_hidden_dim = ff_mlp_hidden_dim

		self.dropout_rate = dropout_rate
		self.use_float64 = use_float64
		self.use_attn_weights_as_feature = use_attn_weights_as_feature

		self.point_embed = pointEncoding.pointEncoding(
			data_dim=self.data_dim,
			num_maps_each_point=self.num_maps_each_point,
			num_samples=self.num_point_encoding_samples) # Phi

		self.attn_layers = torch.nn.ModuleList([
			attn.attnBlock(
				self.attn_head_key_dim,
				self.num_attn_heads,
				self.attn_mlp_hidden_dim,
				self.dropout_rate,
				self.use_float64
			)
			for _ in range(self.num_attn_layers)])

		if self.use_attn_weights_as_feature:
			if self.task == 'split':
				# input_dim = ((self.attn_head_key_dim + self.num_keep_attn_weights + 1) * num_config_points_per_candidate +
				# 			 + self.num_geometric_properties)
				input_dim = ((self.num_keep_attn_weights + 1) * num_config_points_per_candidate +
							 + self.num_geometric_properties)
			else:
				assert self.task == 'subtree'
				input_dim = ((self.attn_head_key_dim + self.num_keep_attn_weights) * num_config_points_per_candidate +
							 + self.num_geometric_properties)
		else:
			input_dim = (self.attn_head_key_dim * num_config_points_per_candidate +
						 # num_config_points_per_candidate * self.data_dim * self.num_maps_each_point
						 + self.num_geometric_properties)


		self.reg = nn.Sequential(
			nn.Linear(input_dim, self.ff_mlp_hidden_dim),
			nn.ReLU(),
			nn.Linear(self.ff_mlp_hidden_dim, self.ff_mlp_hidden_dim),
			nn.ReLU(),
			nn.Linear(self.ff_mlp_hidden_dim, self.ff_mlp_hidden_dim),
			nn.ReLU(),
			nn.Linear(self.ff_mlp_hidden_dim, 1)
			# nn.Tanh()
		)


		if self.use_float64:
			self.double()

	def forward(self, context_rep_points, candi_rep_points, geometric_properties):
		if self.use_attn_weights_as_feature:
			if self.task == 'split':
				return self.forward_split(context_rep_points, candi_rep_points, geometric_properties)
			else:
				return self.forward_subtree(context_rep_points, candi_rep_points, geometric_properties)
		else:
			return self.forward_direct(context_rep_points, candi_rep_points, geometric_properties)

	def forward_direct(self, context_rep_points, candi_rep_points, geometric_properties):
		"""
		:param context_rep_points: Shape `(batch_size, num_subtree_context_rep_points/num_split_context_rep_points, self.data_dim * num_maps_each_point)
		:param candi_rep_points: Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, self.data_dim * num_maps_each_point)
		:param geometric_properties: Shape `(batch_size, max_num_candidates, self.num_geometric_properties)
		:param mask: Shape `(batch_size, max_num_candidates)
		:return:
		"""

		context_rep_points_states = self.point_embed(context_rep_points)
		states = self.point_embed(
			candi_rep_points)  # Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, attn_head_key_dim)

		for i, layer in enumerate(self.attn_layers):
			states, _ = layer(context_rep_points_states, states)

		states = torch.reshape(states, (states.shape[0], self.max_num_candidates, -1))
		# candi_rep_points = torch.reshape(candi_rep_points,
		# 									(candi_rep_points.shape[0], self.max_num_candidates, -1))
		# states = torch.cat([states, candi_rep_points, geometric_properties], dim=-1)
		states = torch.cat([states, geometric_properties], dim=-1)
		scores = self.reg(states)
		scores = scores.squeeze(dim=2)

		min_scores = torch.min(scores, -1)[0]
		min_scores = torch.unsqueeze(min_scores, dim=1)
		final_scores = scores - min_scores

		return final_scores


	def forward_split(self, context_rep_points, candi_rep_points, geometric_properties):
		"""
		:param context_rep_points: Shape `(batch_size, num_subtree_context_rep_points/num_split_context_rep_points, self.data_dim * num_maps_each_point)
		:param candi_rep_points: Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, self.data_dim * num_maps_each_point)
		:param geometric_properties: Shape `(batch_size, max_num_candidates, self.num_geometric_properties)
		:param mask: Shape `(batch_size, max_num_candidates)
		:return:
		"""

		context_rep_points_states = self.point_embed(context_rep_points)
		states = self.point_embed(candi_rep_points) # Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, attn_head_key_dim)

		for i, layer in enumerate(self.attn_layers):
			states, attn_output_weights = layer(context_rep_points_states, states)

		keep_weights = attn_output_weights[:, :, 0:self.num_keep_attn_weights]
		avg_weights = torch.mean(attn_output_weights, dim=-1, keepdim=True)
		# states = torch.concatenate([states, keep_weights, avg_weights], dim=-1)
		states = torch.concatenate([keep_weights, avg_weights], dim=-1)


		states = torch.reshape(states, (states.shape[0], self.max_num_candidates, -1))
		# candi_rep_points = torch.reshape(candi_rep_points,
		# 									(candi_rep_points.shape[0], self.max_num_candidates, -1))
		# states = torch.cat([states, candi_rep_points, geometric_properties], dim=-1)
		states = torch.cat([states, geometric_properties], dim=-1)
		scores = self.reg(states)
		scores = scores.squeeze(dim=2)

		min_scores = torch.min(scores, -1)[0]
		min_scores = torch.unsqueeze(min_scores, dim=1)
		final_scores = scores - min_scores

		return final_scores

	def forward_subtree(self, context_rep_points, candi_rep_points, geometric_properties):
		"""
		:param context_rep_points: Shape `(batch_size, num_subtree_context_rep_points/num_split_context_rep_points, self.data_dim * num_maps_each_point)
		:param candi_rep_points: Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, self.data_dim * num_maps_each_point)
		:param geometric_properties: Shape `(batch_size, max_num_candidates, self.num_geometric_properties)
		:param mask: Shape `(batch_size, max_num_candidates)
		:return:
		"""

		context_rep_points_states = self.point_embed(context_rep_points)
		states = self.point_embed(
			candi_rep_points)  # Shape `(batch_size, max_num_candidates * num_config_points_per_candidate, attn_head_key_dim)

		for i, layer in enumerate(self.attn_layers):
			states, attn_output_weights = layer(context_rep_points_states, states)

		states = torch.concatenate([states, attn_output_weights], dim=-1)

		states = torch.reshape(states, (states.shape[0], self.max_num_candidates, -1))
		# candi_rep_points = torch.reshape(candi_rep_points,
		# 									(candi_rep_points.shape[0], self.max_num_candidates, -1))
		# states = torch.cat([states, candi_rep_points, geometric_properties], dim=-1)
		states = torch.cat([states, geometric_properties], dim=-1)
		scores = self.reg(states)
		scores = scores.squeeze(dim=2)

		min_scores = torch.min(scores, -1)[0]
		min_scores = torch.unsqueeze(min_scores, dim=1)
		final_scores = scores - min_scores

		return final_scores
