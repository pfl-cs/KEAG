import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial
from torch.optim.lr_scheduler import LambdaLR

import sys

# sys.path.append("../../")
from pytorch_lightning.utilities import rank_zero_info


def compute_loss(loss_fun, pred, true, task_type=None):
	bce_loss = nn.BCEWithLogitsLoss()
	mse_loss = nn.MSELoss()

	# default manipulation for pred and true
	# can be skipped if special loss computation is needed


	# pred = pred.squeeze(-1) if pred.ndim > 1 else pred
	# true = true.squeeze(-1) if true.ndim > 1 else true

	if task_type is None:
		if loss_fun == 'cross_entropy':
			# multiclass
			if pred.ndim > 1 and true.ndim == 1:
				pred = F.log_softmax(pred, dim=-1)
				return F.nll_loss(pred, true), pred
			# binary or multilabel
			else:
				true = true.float()
				return bce_loss(pred, true), torch.sigmoid(pred)
		elif loss_fun == 'mse':
			true = true.float()
			return mse_loss(pred, true), pred
		elif loss_fun == 'mse_two_side':
			true = true.float()
			true_reverse = (1 - true).float()
			norm1 = torch.norm(pred - true, dim=1) / true.shape[1]
			norm2 = torch.norm(pred - true_reverse, dim=1) / true.shape[1]
			norm = torch.minimum(norm1, norm2)
			loss = torch.mean(norm)
			return loss, pred
		elif loss_fun == 'kl_div' or loss_fun == 'kl_divergence':
			# print(f'++++++++{loss_fun}')
			kl_loss = nn.KLDivLoss(reduction="batchmean")
			# input = F.log_softmax(pred, dim=-1)
			# Sample a batch of distributions. Usually this would come from the dataset
			target = F.softmax(true, dim=1)
			loss = kl_loss(pred, target)

			return loss, pred
		else:
			raise ValueError('Loss func {} not supported'.format(
				loss_fun))
	else:
		if task_type == 'classification_multi':
			pred = F.log_softmax(pred, dim=-1)
			return F.nll_loss(pred, true), pred
		elif 'classification' in task_type and 'binary' in task_type:
			true = true.float()
			return bce_loss(pred, true), torch.sigmoid(pred)
		elif task_type == 'regression':
			true = true.float()
			# return mse_loss(torch.exp(pred), torch.exp(true)), pred
			# print(f'pred.shape = {pred.shape}')
			# print(f'true.shape = {true.shape}')
			return mse_loss(pred, true), pred
		else:
			raise ValueError('Task type {} not supported'.format(task_type))


# TODO: some parameters could be further refactored
def create_optimizer(cfg, params):
	r"""Creates a config-driven optimizer."""
	params = filter(lambda p: p.requires_grad, params)

	if cfg.optim.optimizer == 'adam':
		optimizer = optim.Adam(params,
							   lr=cfg.optim.base_lr,
							   weight_decay=cfg.optim.weight_decay)
	elif cfg.optim.optimizer == 'adamw':
		optimizer = optim.AdamW(params,
							   lr=cfg.optim.base_lr,
							   weight_decay=cfg.optim.weight_decay)
	elif cfg.optim.optimizer == 'sgd':
		optimizer = optim.SGD(params,
							  lr=cfg.optim.base_lr,
							  momentum=cfg.optim.momentum,
							  weight_decay=cfg.optim.weight_decay)
	else:
		raise ValueError('Optimizer {} not supported'.format(
			cfg.optim.optimizer))

	return optimizer


def create_scheduler(cfg, optimizer):
	r"""Creates a config-driven learning rate scheduler."""
	if cfg.optim.scheduler == 'none':
		scheduler = optim.lr_scheduler.StepLR(optimizer,
											  step_size=cfg.optim.max_epoch +
														1)
	elif cfg.optim.scheduler == 'step':
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
												   milestones=cfg.optim.steps,
												   gamma=cfg.optim.lr_decay)
	elif cfg.optim.scheduler == 'cos':
		scheduler = optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=cfg.optim.max_epoch)
	else:
		raise ValueError('Scheduler {} not supported'.format(
			cfg.optim.scheduler))
	return scheduler


def get_schedule_fn(scheduler, num_training_steps):
	"""Returns a callable scheduler_fn(optimizer).
	Todo: Sanitize and unify these schedulers...
	"""
	if scheduler == "cosine-decay":
		scheduler_fn = partial(
			torch.optim.lr_scheduler.CosineAnnealingLR,
			T_max=num_training_steps,
			eta_min=0.0,
		)
	elif scheduler == "one-cycle":  # this is a simplified one-cycle
		scheduler_fn = partial(
			get_one_cycle,
			num_training_steps=num_training_steps,
		)
	else:
		raise ValueError(f"Invalid schedule {scheduler} given.")
	return scheduler_fn


def get_one_cycle(optimizer, num_training_steps):
	"""Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""

	def lr_lambda(current_step):
		if current_step < num_training_steps / 2:
			return float(current_step / (num_training_steps / 2))
		else:
			return float(2 - current_step / (num_training_steps / 2))

	return LambdaLR(optimizer, lr_lambda, -1)


def configure_optimizers(cfg, params, total_num_training_steps=None):
	rank_zero_info('Parameters: %d' % sum([p.numel() for p in params]))
	rank_zero_info('Training steps: %d' % total_num_training_steps)

	if cfg.optim.lr_scheduler == "constant":
		return torch.optim.AdamW(
			params, lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)

	else:
		optimizer = torch.optim.AdamW(
			params, lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
		scheduler = get_schedule_fn(cfg.optim.lr_scheduler, total_num_training_steps)(
			optimizer)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "step",
			},
		}
