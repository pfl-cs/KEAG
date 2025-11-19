import os
import sys
import math
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from nn_blocks import pointEncoding
from nn_blocks import attn
from utils import data_utils, ckpt_utils, train_utils, FileViewer
import config
from .KEAG import KEAG
from .simpleScoreNet import simpleScoreNet


class rtreeModel(nn.Module):
	def __init__(self,
				 cfg):
		super(rtreeModel, self).__init__()
		if cfg.task == 'subtree':
			self.loss_func = cfg.subtree_model.loss_func
			self.ckpt_dir = cfg.subtree.ckpt_dir
			if cfg.model_type == 'KEAG':
				self.model_name = 'subtree'
				self.model = KEAG(
					task=cfg.task,
					data_dim=cfg.dataset.dim,
					max_num_candidates=cfg.subtree.max_num_subtrees,
					num_keep_attn_weights=cfg.subtree.num_subtree_context_rep_points,
					num_config_points_per_candidate=cfg.subtree.num_candi_rep_points_per_subtree,
					num_maps_each_point=cfg.subtree.num_maps_each_point,
					num_geometric_properties=cfg.subtree.num_geometric_properties,
					num_point_encoding_samples=cfg.subtree_model.num_point_encoding_samples,
					num_attn_layers=cfg.subtree_model.num_attn_layers,
					num_attn_heads=cfg.subtree_model.num_attn_heads,
					attn_mlp_hidden_dim=cfg.subtree_model.attn_mlp_hidden_dim,
					attn_mlp_output_dim=cfg.subtree_model.attn_mlp_output_dim,
					ff_mlp_num_layers=cfg.subtree_model.ff_mlp_num_layers,
					ff_mlp_hidden_dim=cfg.subtree_model.ff_mlp_hidden_dim,
					dropout_rate=cfg.subtree_model.dropout_rate,
					use_float64=cfg.subtree_model.use_float64
				)
				cfg.subtree_model.attn_mlp_hidden_dim = self.model.attn_mlp_hidden_dim
			else:
				if cfg.model_type == 'wophi' or cfg.model_type == 'simple':
					self.model_name = 'wophi_subtree'
					use_spatial_points = True
				else:
					assert cfg.model_type == 'wopoint' or cfg.model_type == 'wopoints' or cfg.model_type == 'simplest'
					self.model_name = 'wopoint_subtree'
					use_spatial_points = False
				self.model = simpleScoreNet(
					data_dim=cfg.dataset.dim,
					max_num_candidates=cfg.subtree.max_num_subtrees,
					num_config_points_per_candidate=cfg.subtree.num_candi_rep_points_per_subtree,
					num_maps_each_point=cfg.subtree.num_maps_each_point,
					num_geometric_properties=cfg.subtree.num_geometric_properties,
					ff_mlp_num_layers=cfg.subtree_model.ff_mlp_num_layers,
					ff_mlp_hidden_dim=cfg.subtree_model.ff_mlp_hidden_dim,
					use_spatial_points=use_spatial_points,
					dropout_rate=cfg.subtree_model.dropout_rate,
					use_float64=cfg.subtree_model.use_float64
				)
				# cfg.subtree_model.attn_mlp_hidden_dim = self.model.attn_mlp_hidden_dim
		else:
			self.loss_func = cfg.split_model.loss_func
			self.ckpt_dir = cfg.split.ckpt_dir
			if cfg.model_type == 'KEAG':
				self.model_name = 'split'
				self.model = KEAG(
					task=cfg.task,
					data_dim=cfg.dataset.dim,
					max_num_candidates=cfg.split.max_num_splits,
					num_keep_attn_weights=cfg.split.num_points_from_parent_node,
					num_config_points_per_candidate=cfg.split.num_candi_rep_points_per_split,
					num_maps_each_point=cfg.split.num_maps_each_point,
					num_geometric_properties=cfg.split.num_geometric_properties,
					num_point_encoding_samples=cfg.split_model.num_point_encoding_samples,
					num_attn_layers=cfg.split_model.num_attn_layers,
					num_attn_heads=cfg.split_model.num_attn_heads,
					attn_mlp_hidden_dim=cfg.split_model.attn_mlp_hidden_dim,
					attn_mlp_output_dim=cfg.split_model.attn_mlp_output_dim,
					ff_mlp_num_layers=cfg.split_model.ff_mlp_num_layers,
					ff_mlp_hidden_dim=cfg.split_model.ff_mlp_hidden_dim,
					dropout_rate=cfg.split_model.dropout_rate,
					use_float64=cfg.split_model.use_float64
				)
				cfg.split_model.attn_mlp_hidden_dim = self.model.attn_mlp_hidden_dim
			else:
				if cfg.model_type == 'wophi' or cfg.model_type == 'simple':
					self.model_name = 'wophi_split'
					use_spatial_points = True
				else:
					assert cfg.model_type == 'wopoint' or cfg.model_type == 'wopoints' or cfg.model_type == 'simplest'
					self.model_name = 'wopoint_split'
					use_spatial_points = False

				self.model = simpleScoreNet(
					data_dim=cfg.dataset.dim,
					max_num_candidates=cfg.split.max_num_splits,
					num_config_points_per_candidate=cfg.split.num_candi_rep_points_per_split,
					num_maps_each_point=cfg.split.num_maps_each_point,
					num_geometric_properties=cfg.split.num_geometric_properties,
					ff_mlp_num_layers=cfg.split_model.ff_mlp_num_layers,
					ff_mlp_hidden_dim=cfg.split_model.ff_mlp_hidden_dim,
					use_spatial_points=use_spatial_points,
					dropout_rate=cfg.split_model.dropout_rate,
					use_float64=cfg.split_model.use_float64
				)
				# cfg.split_model.attn_mlp_hidden_dim = self.model.attn_mlp_hidden_dim

		self.ckpt_fname = config.get_model_ckpt_fname(cfg, self.model_name)
		self.jit_fname = f'{self.model_name}'

	def forward(self, context_rep_points, candi_rep_points, geometric_properties):
		return self.model(context_rep_points, candi_rep_points, geometric_properties)

	def training_step(self, context_rep_points, candi_rep_points, geometric_properties, scores, mask):
		pred = self.forward(
			context_rep_points,
			candi_rep_points,
			geometric_properties
		)

		if self.loss_func == 'kl_div' or self.loss_func == 'kl_divergence':
			kl_loss = nn.KLDivLoss(reduction="batchmean")
			# input should be a distribution in the log space
			input = F.log_softmax(pred * mask, dim=-1)
			# Sample a batch of distributions. Usually this would come from the dataset
			target = F.softmax(scores * mask, dim=1)
			loss = kl_loss(input, target)
		else:
			loss = F.mse_loss(pred * mask, scores * mask)

		# self.log("train/loss", loss)
		return loss


	def test_step(self, context_rep_points, candi_rep_points, geometric_properties, mask):
		with torch.no_grad():
			return self._test_step(context_rep_points, candi_rep_points, geometric_properties, mask)

	def _test_step(self, context_rep_points, candi_rep_points, geometric_properties, mask):
		pred = self.forward(
			context_rep_points,
			candi_rep_points,
			geometric_properties
		)

		if self.loss_func == 'kl_div' or self.loss_func == 'kl_divergence':
			return F.log_softmax(pred * mask, dim=-1)
		else:
			return pred * mask

	def save_estimator_for_cpp(self, context_rep_points, candi_rep_points, geometric_properties, device):
		self.model.eval()
		traced_script_estimator = torch.jit.trace(self.model, example_inputs=(
			context_rep_points.to(device), candi_rep_points.to(device), geometric_properties.to(device)))
		ckpt_utils.save_traced_script(self.ckpt_dir, self.jit_fname, traced_script_estimator)
