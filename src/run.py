import os.path
import torch
import numpy as np
import config
from utils import data_utils, ckpt_utils, train_utils, FileViewer
from data_process import loader
import config
import os
from tqdm import tqdm
from model import rtreeModel
import math
import sys
import scipy

def eval_model(device, loader, model, example_inputs=None):
	model.eval()
	count = 0
	avg_loss = 0

	preds = []
	gts = []
	loss_func = model.loss_func
	with torch.no_grad():
		task_type = None
		for _, (eval_context_rep_points, eval_candi_rep_points, eval_geometric_properties, eval_scores, eval_mask) in enumerate(loader):
			if example_inputs is not None:
				if len(example_inputs) == 0:
					example_inputs.append(eval_context_rep_points[0:1, :, :])
					example_inputs.append(eval_candi_rep_points[0:1, :, :])
					example_inputs.append(eval_geometric_properties[0:1, :])


			eval_context_rep_points = eval_context_rep_points.to(device)
			eval_candi_rep_points = eval_candi_rep_points.to(device)
			eval_geometric_properties = eval_geometric_properties.to(device)
			eval_scores = eval_scores.to(device)
			eval_mask = eval_mask.to(device)
			# optimizer.zero_grad()
			pred = model.test_step(eval_context_rep_points, eval_candi_rep_points, eval_geometric_properties, eval_mask)

			eval_scores *= eval_mask
			batch_size = eval_scores.shape[0]

			loss, _ = train_utils.compute_loss(loss_func, pred, eval_scores, task_type)

			avg_loss += loss.detach().cpu() * batch_size
			count += batch_size
			# _preds.append(pred.detach().cpu())
			gt = eval_scores.detach().cpu().numpy()
			pred = pred.detach().cpu().numpy()
			gts.append(gt)
			preds.append(pred)
	# scheduler.step()
	avg_loss /= count
	preds = np.concatenate(preds, axis=0)
	gts = np.concatenate(gts, axis=0)

	if loss_func == 'kl_div' or loss_func == 'kl_divergence':
		gts = scipy.special.softmax(gts, axis=-1)
		preds = np.exp(preds)

	return avg_loss, preds


def train_epoch(device, loader, model, optimizer, scheduler):
	model.train()
	count = 0
	avg_loss = 0

	loop_train = tqdm(loader, leave=False)
	for (train_context_rep_points, train_candi_rep_points, train_geometric_properties, train_scores, train_mask) in loop_train:
		train_context_rep_points = train_context_rep_points.to(device)
		train_candi_rep_points = train_candi_rep_points.to(device)
		train_geometric_properties = train_geometric_properties.to(device)
		train_scores = train_scores.to(device)
		train_mask = train_mask.to(device)
		batch_size = train_scores.shape[0]

		optimizer.zero_grad()
		loss = model.training_step(train_context_rep_points, train_candi_rep_points, train_geometric_properties, train_scores, train_mask)
		loss.backward()
		optimizer.step()
		avg_loss += loss.detach().cpu() * batch_size

		count += batch_size
	scheduler.step()
	avg_loss /= count

	return avg_loss

def train(device, train_loader, eval_loader,
		  model, optimizer, scheduler,
		  MAX_CKPT_KEEP_NUM, max_epoch, start_epoch=0, min_loss=math.inf):
	loss_func = model.loss_func
	ckpt_dir = model.ckpt_dir
	ckpt_fname = model.ckpt_fname
	if not os.path.exists(ckpt_dir):
		print(f"[INFO] creating the directory {ckpt_dir} to save checkpoints")
		os.makedirs(ckpt_dir)

	example_inputs = []

	for epoch in range(max_epoch):
		train_loss = train_epoch(device, train_loader, model, optimizer, scheduler)
		train_loss = round(float(train_loss), 5)
		if epoch < 5:
			continue

		val_loss, _ = eval_model(device, eval_loader, model, example_inputs)
		if val_loss < min_loss:
			min_loss = val_loss
			ckpt_utils.save_ckpt(ckpt_dir, ckpt_fname, epoch + start_epoch, model, optimizer,
								 MAX_CKPT_KEEP_NUM, train_loss=train_loss, val_loss=val_loss)

			model.save_estimator_for_cpp(example_inputs[0], example_inputs[1], example_inputs[2], device)
			min_loss = round(float(min_loss), 5)
		val_loss = round(float(val_loss), 5)

		print(f'epoch-{epoch + start_epoch}, train_loss = {train_loss}, val_loss = {val_loss}, min_loss = {min_loss}')

if __name__ == '__main__':
	cfg = config.getConfigs()
	if torch.cuda.is_available():
		device = torch.device(f'cuda:{cfg.run.gpu}')
		torch.cuda.empty_cache()
	else:
		device = torch.device(f'cpu')
	print(f'device = {device}')

	if cfg.task == 'split':
		(train_data, validation_data) = loader.load_split_data(cfg)
	else:
		(train_data, validation_data) = loader.load_subtree_data(cfg)

	train_loader, validation_loader = loader.create_loaders(train_data, validation_data, cfg.run.batch_size)

	model = rtreeModel.rtreeModel(cfg)
	loss_func = model.loss_func
	ckpt_dir = model.ckpt_dir
	ckpt_fname = model.ckpt_fname

	model_loaded = False
	MAX_CKPT_KEEP_NUM = cfg.run.MAX_CKPT_KEEP_NUM

	min_loss = math.inf
	model_start_epoch = 0
	if cfg.run.eval_model:
		print(f'ckpt_dir = {ckpt_dir}, ckpt_fname = {ckpt_fname}')
		model, model_start_epoch, model_best_loss =\
			ckpt_utils.load_model(ckpt_dir, ckpt_fname, model, device=device)

		if model_start_epoch >= 0:
			print("Finished loading model.")
			eval_model(device, loader=train_loader, model=model)
			model_loaded = True
		else:
			model_start_epoch = 0
			print("There is no available model.")

	if cfg.run.train_model:
		model.to(device)
		optimizer = train_utils.create_optimizer(cfg, model.parameters())
		scheduler = train_utils.create_scheduler(cfg, optimizer)

		max_epoch = cfg.optim.max_epoch
		train(device, train_loader, validation_loader,
			  model, optimizer, scheduler,
			  MAX_CKPT_KEEP_NUM, max_epoch,
			  start_epoch=model_start_epoch, min_loss=min_loss)

# python run.py --task split --train_model True --data GAU --data_size 2
