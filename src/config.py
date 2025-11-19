import os
import argparse
import math
from distutils.util import strtobool
from yacs.config import CfgNode as CN
import math
from pathlib import Path


def init_cfg():
    cfg = CN()

    cfg.task = 'split' # 'split' or subtree
    cfg.model_type = 'KEAG' # 'KEAG' or 'simple' or simplest

    cfg.project_root = None
    cfg.data_root = None
    # Dataset options
    cfg.dataset = CN()
    cfg.dataset.name = 'guassian'

    cfg.dataset.dim = 2
    cfg.dataset.data_size = 25  # in millions
    cfg.dataset.use_across = False

    # ----------------------------------------------------------------------- #
    # Split options
    # ----------------------------------------------------------------------- #
    cfg.split = CN()
    cfg.split.split_records_fname = 'split_records.bin'
    cfg.split.normalization_infos_fname = 'split_norm_infos.bin'
    cfg.split.num_geometric_properties = 5
    cfg.split.num_maps_each_point = 2
    cfg.split.num_points_from_parent_node = 9
    cfg.split.num_split_context_rep_points = 60
    cfg.split.num_candi_rep_points_per_split = 8
    cfg.split.max_num_splits = 52

    cfg.split.ckpt_dir = ''

    # ----------------------------------------------------------------------- #
    # Subtree options
    # ----------------------------------------------------------------------- #
    cfg.subtree = CN()
    cfg.subtree.subtree_records_fname = 'subtree_records.bin'
    cfg.subtree.normalization_infos_fname = 'subtree_norm_infos.bin'
    cfg.subtree.num_geometric_properties = 4
    cfg.subtree.num_maps_each_point = 2
    cfg.subtree.num_subtree_context_rep_points = 4
    cfg.subtree.num_candi_rep_points_per_subtree = 4
    cfg.subtree.max_num_subtrees = 52

    # ----------------------------------------------------------------------- #
    # Running options
    # ----------------------------------------------------------------------- #
    cfg.run = CN()

    # Training (and validation) pipeline mode
    cfg.run.exp_dir = ''

    # TODO: Params that need to be overwrritten
    cfg.run.batch_size = 128  # 16
    cfg.run.gpu = 0
    cfg.run.train_model = True
    cfg.run.force_retrain = False
    cfg.run.eval_model = False
    cfg.run.MAX_CKPT_KEEP_NUM = 1
    cfg.run.sparse_factor = 1

    # ----------------------------------------------------------------------- #
    # Parameters wrt the split model
    # ----------------------------------------------------------------------- #
    cfg.split_model = CN()
    cfg.split_model.use_float64 = False
    cfg.split_model.dropout_rate = 0
    cfg.split_model.num_point_encoding_samples = 64
    cfg.split_model.loss_func = 'kl_div' # 'mse' or 'kl_div'

    cfg.split_model.num_attn_layers = 3
    cfg.split_model.num_attn_heads = 8
    cfg.split_model.attn_head_key_dim = 512  # 1024,
    cfg.split_model.attn_mlp_hidden_dim = 512  # 1024
    cfg.split_model.attn_mlp_output_dim = 512  # 1024
    cfg.split_model.ff_mlp_num_layers = 3
    cfg.split_model.ff_mlp_hidden_dim = 512  # 1024

    # ----------------------------------------------------------------------- #
    # Parameters wrt the subtree model
    # ----------------------------------------------------------------------- #
    cfg.subtree_model = CN()
    cfg.subtree_model.use_float64 = False
    cfg.subtree_model.dropout_rate = 0
    cfg.subtree_model.num_point_encoding_samples = 64
    cfg.subtree_model.loss_func = 'kl_div'  # 'mse' or 'kl_div'

    cfg.subtree_model.num_attn_layers = 3
    cfg.subtree_model.num_attn_heads = 8
    cfg.subtree_model.attn_head_key_dim = 512  # 1024,
    cfg.subtree_model.attn_mlp_hidden_dim = 512  # 1024
    cfg.subtree_model.attn_output_hidden_dim = 512  # 1024
    cfg.subtree_model.ff_mlp_num_layers = 3
    cfg.subtree_model.ff_mlp_hidden_dim = 512  # 1024

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()
    cfg.optim.lr_scheduler = 'constant'
    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'
    # Base learning rate
    cfg.optim.base_lr = 1e-3  # 0.01
    # L2 regularization
    cfg.optim.weight_decay = 5e-4
    # SGD momentum
    cfg.optim.momentum = 0.9
    # scheduler: none, step, cos
    cfg.optim.scheduler = 'none'
    # cfg.optim.scheduler = 'step'
    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]
    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1
    # Maximal number of epochs
    cfg.optim.max_epoch = 30

    cfg.rtree = CN()
    cfg.rtree.overflow_threshold = 65

    return cfg


def get_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, default='split', help='split or subtree')
    parser.add_argument('--model_type', type=str, default='KEAG', help='KEAG or simple')

    # params of cfg.dataset
    parser.add_argument('--data', type=str, default='guassian', help='data distribution(GAU, Zipf, UNI, CHN or IND)')
    parser.add_argument('--data_size', type=int, default=1, help='data size (in millions)')
    parser.add_argument('--use_across', type=lambda x: bool(strtobool(x)), default=False, help='')
    # params of cfg.optim
    parser.add_argument('--loss_func', type=str, default='kl_div', help='')

    # params of cfg.run
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--train_model', type=lambda x: bool(strtobool(x)), default=False, help='')
    parser.add_argument('--eval_model', type=lambda x: bool(strtobool(x)), default=False, help='')

    # params of cfg.optim
    parser.add_argument('--base_lr', type=float, default=1e-3, help='')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
    parser.add_argument('--max_epoch', type=int, default=30, help='')

    args = parser.parse_args()
    return args


def overwrite_from_args(args, cfg):
    # overwrite some default config parameters from the arg-parser
    cfg.task = args.task
    cfg.model_type = args.model_type
    cfg.dataset.name = args.data
    cfg.dataset.data_size = args.data_size
    cfg.dataset.use_across = args.use_across
    cfg.split_model.loss_func = args.loss_func
    cfg.subtree_model.loss_func = args.loss_func
    cfg.run.gpu = args.gpu
    cfg.run.train_model = args.train_model
    cfg.run.eval_model = args.eval_model
    if args.base_lr > 0:
        cfg.optim.base_lr = args.base_lr
    if args.weight_decay > 0:
        cfg.optim.weight_decay = args.weight_decay
    if args.max_epoch > 0:
        cfg.optim.max_epoch = args.max_epoch


def set_cfg(cfg):
    assert cfg is not None
    cfg.data_root = os.path.join(cfg.project_root, 'data')
    if cfg.dataset.use_across:
        data_info = f'{cfg.dataset.name}/across'
    else:
        data_info = f'{cfg.dataset.name}/{cfg.dataset.data_size}M'

    cfg.split.ckpt_dir = os.path.join(cfg.project_root, f'ckpt/split/{data_info}')
    cfg.subtree.ckpt_dir = os.path.join(cfg.project_root, f'ckpt/subtree/{data_info}')


def getConfigs():
    cfg = init_cfg()
    args = get_arg_parser()
    overwrite_from_args(args, cfg)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = Path(cur_dir).parent.absolute()
    cfg.project_root = str(project_root)

    set_cfg(cfg)

    return cfg


def get_data_dirs(cfg):
    split_records_paths = []
    split_normalization_infos_paths = []
    subtree_records_paths = []
    subtree_normalization_infos_paths = []

    if cfg.dataset.use_across:
        data_sizes = [2, 10, 25]
        # data_sizes = [2, 10]
        tag = 'across_'
    else:
        data_sizes = [cfg.dataset.data_size]
        tag = ''

    for data_size in data_sizes:
        data_info = f'{cfg.dataset.name}/{data_size}M'
        data_dir = os.path.join(cfg.data_root, data_info)

        split_records_path = os.path.join(data_dir, cfg.split.split_records_fname)
        subtree_records_path = os.path.join(data_dir, cfg.subtree.subtree_records_fname)
        split_records_paths.append(split_records_path)
        subtree_records_paths.append(subtree_records_path)

    if cfg.dataset.use_across:
        data_sizes.extend([50, 100])

    for data_size in data_sizes:
        data_info = f'{cfg.dataset.name}/{data_size}M'
        data_dir = os.path.join(cfg.data_root, data_info)
        split_normalization_infos_path = os.path.join(data_dir, f'{tag}{cfg.split.normalization_infos_fname}')
        split_normalization_infos_paths.append(split_normalization_infos_path)
        subtree_normalization_infos_path = os.path.join(data_dir, f'{tag}{cfg.subtree.normalization_infos_fname}')
        subtree_normalization_infos_paths.append(subtree_normalization_infos_path)

    return split_records_paths, split_normalization_infos_paths, subtree_records_paths, subtree_normalization_infos_paths



def get_model_ckpt_fname(cfg, model_name):
    assert model_name is not None
    if cfg.task == 'split':
        s1 = f'{cfg.split_model.num_attn_layers}_{cfg.split_model.num_point_encoding_samples}'  # {cfg.split_model.ff_mlp_num_layers}'
        s2 = f'{int(math.log2(cfg.split_model.attn_mlp_hidden_dim + 1))}_{int(math.log2(cfg.split_model.ff_mlp_hidden_dim + 1))}'
    else:
        s1 = f'{cfg.subtree_model.num_attn_layers}_{cfg.subtree_model.num_point_encoding_samples}'  # {cfg.subtree_model.ff_mlp_num_layers}'
        s2 = f'{int(math.log2(cfg.subtree_model.attn_mlp_hidden_dim + 1))}_{int(math.log2(cfg.subtree_model.ff_mlp_hidden_dim + 1))}'
    model_ckpt_fname = f'{model_name}_{s1}_{s2}'
    return model_ckpt_fname