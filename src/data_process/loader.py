import copy
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import scipy
# sys.path.append("../")
from utils import data_utils
import config
import struct

class myDataset(Dataset):
    def __init__(self, data):
        self.feature_points = data[0]
        self.candi_rep_points = data[1]
        self.geometric_properties = data[2]
        self.scores = data[3]
        self.mask = data[4]

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, idx):
        return self.feature_points[idx], self.candi_rep_points[idx], self.geometric_properties[idx], self.scores[idx], self.mask[idx]

def create_loaders(train_data, validation_data, batch_size):
    train_dataset = myDataset(train_data)
    validation_dataset = myDataset(validation_data)

    loader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False)

    loader_validation = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    return loader_train, loader_validation

def batch_repeat(data, repeats):
    """
    Args:
        data: N * M array
    Returns:
        data: N * repeats * M array
    """
    data = np.expand_dims(data, axis=1)
    data = np.repeat(data, repeats, axis=1)
    return data

def scale_and_expand_points(points, lbds_local, diff_local, lbds_global, diff_global):
    assert len(points.shape) == 3
    points_local = (points - lbds_local) / diff_local
    points_global = (points - lbds_global) / diff_global

    points = np.zeros(shape=(points.shape[0], points.shape[1], points.shape[2] * 2), dtype=points.dtype)
    points[:, :, 0:points.shape[2]:2] = points_local
    points[:, :, 1:points.shape[2]:2] = points_global
    # points = points * 2.0 - 1
    return points

def normalize_features(train_features, validation_features_list, reshape_train_features=True):
    _train_shape = train_features.shape
    if reshape_train_features:
        train_features = np.reshape(train_features, [-1, _train_shape[-1]])
    train_std = train_features.std(axis=0)

    std_zero_idxes = np.where(train_std == 0)[0]
    train_mean = train_features.mean(axis=0)
    train_std[std_zero_idxes] = 1
    train_features = (train_features - train_mean) / train_std

    num = len(validation_features_list)
    for i in range(num):
        validation_features = validation_features_list[i]
        _validation_shape = validation_features.shape
        # validation_features = validation_features[:, nonzero_idxes]
        validation_features = np.reshape(validation_features, [-1, _validation_shape[-1]])
        validation_features = (validation_features - train_mean) / train_std
        validation_features_list[i] = np.reshape(validation_features, _validation_shape)

    if reshape_train_features:
        train_features = np.reshape(train_features, _train_shape)
    return train_mean, train_std, train_features, validation_features_list

def process_diff(diff):
    zero_idxes = np.where(diff == 0)
    return diff

def process_subtree_data(cfg, data):
    (dim, lbds, ubds, _, _,
     num_subtree_context_rep_points, num_candi_rep_points_per_subtree,
     num_geometric_properties, data_num_subtrees,
     all_parent_mbr_mins, all_parent_mbr_maxs,
     all_subtree_context_rep_points, all_subtree_candi_rep_points, all_geometric_properties,
     all_subtree_scores) = data

    all_subtree_scores = all_subtree_scores[:, 0:cfg.subtree.max_num_subtrees]
    all_subtree_candi_rep_points = all_subtree_candi_rep_points[:,
                                      0:cfg.subtree.max_num_subtrees * num_candi_rep_points_per_subtree, :]
    all_geometric_properties = all_geometric_properties[:, 0:cfg.subtree.max_num_subtrees, :]

    lbds_local = batch_repeat(all_parent_mbr_mins, num_subtree_context_rep_points)
    ubds_local = batch_repeat(all_parent_mbr_maxs, num_subtree_context_rep_points)
    diff_local = ubds_local - lbds_local

    lbds_global = lbds
    ubds_global = ubds
    diff_global = ubds_global - lbds_global

    all_subtree_context_rep_points = scale_and_expand_points(all_subtree_context_rep_points, lbds_local, diff_local,
                                                        lbds_global, diff_global)

    num_candi_rep_points_all_subtrees = cfg.subtree.max_num_subtrees * cfg.num_candi_rep_points_per_subtree
    if num_candi_rep_points_all_subtrees <= num_subtree_context_rep_points:
        lbds_local = lbds_local[:, 0:num_candi_rep_points_all_subtrees, :]
        diff_local = diff_local[:, 0:num_candi_rep_points_all_subtrees, :]
    else:
        lbds_local = batch_repeat(all_parent_mbr_mins, num_candi_rep_points_all_subtrees)
        ubds_local = batch_repeat(all_parent_mbr_maxs, num_candi_rep_points_all_subtrees)
        diff_local = ubds_local - lbds_local

    all_subtree_candi_rep_points = scale_and_expand_points(all_subtree_candi_rep_points, lbds_local, diff_local,
                                                              lbds_global, diff_global)

    # subtree_point_max = np.max(all_subtree_candi_rep_points)
    # subtree_point_min = np.min(all_subtree_candi_rep_points)
    # subtree_point_mean = np.mean(all_subtree_candi_rep_points)

    all_points = [np.reshape(all_subtree_context_rep_points, (-1, all_subtree_context_rep_points.shape[-1]))]
    data_size = all_subtree_context_rep_points.shape[0]
    assert all_subtree_candi_rep_points.shape[1] == num_candi_rep_points_all_subtrees

    all_geometric_properties_wo_padding = []
    for i in range(data_size):
        num_subtrees = data_num_subtrees[i]
        subtree_candi_rep_points_i = all_subtree_candi_rep_points[i,
                                        0:num_subtrees * num_candi_rep_points_per_subtree, :]
        assert len(subtree_candi_rep_points_i.shape) == 2
        # if subtree_candi_rep_points_i.shape[0] != num_subtrees * num_candi_rep_points_per_subtree or subtree_candi_rep_points_i.shape[1] != dim * cfg.subtree.num_maps_each_point:
        #     print(
        #         f'i = {i}, all_subtree_candi_rep_points.shape: {all_subtree_candi_rep_points.shape}, num_subtrees = {num_subtrees}, subtree_candi_rep_points_i.shape: {subtree_candi_rep_points_i.shape}, {num_subtrees * num_candi_rep_points_per_subtree}, {dim * cfg.subtree.num_maps_each_point}')
        assert subtree_candi_rep_points_i.shape[0] == num_subtrees * cfg.subtree.num_candi_rep_points_per_subtree and \
               subtree_candi_rep_points_i.shape[1] == dim * cfg.subtree.num_maps_each_point
        all_points.append(subtree_candi_rep_points_i)

        geometric_properties_i = all_geometric_properties[i, 0:num_subtrees, :]
        all_geometric_properties_wo_padding.append(geometric_properties_i)

    all_points = np.concatenate(all_points, axis=0)
    all_geometric_properties_wo_padding = np.concatenate(all_geometric_properties_wo_padding, axis=0)

    all_masks = np.zeros(shape=[data_size, cfg.subtree.max_num_subtrees], dtype=all_subtree_scores.dtype)
    for i in range(data_size):
        num_subtrees = data_num_subtrees[i]
        all_masks[i, 0:num_subtrees] = 1

    return all_subtree_context_rep_points, all_subtree_candi_rep_points, all_geometric_properties, all_subtree_scores, all_masks, all_points, all_geometric_properties_wo_padding

def load_subtree_data(cfg):
    _, _, subtree_records_paths, subtree_normalization_infos_paths = config.get_data_dirs(cfg)

    num_datas = len(subtree_records_paths)
    all_datas = []
    max_num_subtrees = 0

    for subtree_records_path in subtree_records_paths:
        all_data_i = data_utils.load_subtree_records(subtree_records_path)
        all_datas.append(all_data_i)
        (dim, lbds, ubds, max_num_subtrees_i, _,
         num_subtree_context_rep_points, num_candi_rep_points_per_subtree,
         num_geometric_properties, data_num_subtrees,
         all_parent_mbr_mins, all_parent_mbr_maxs,
         all_subtree_context_rep_points, all_subtree_candi_rep_points, all_geometric_properties,
         all_subtree_scores) = all_data_i

        if max_num_subtrees_i > max_num_subtrees:
            max_num_subtrees = max_num_subtrees_i

        cfg.subtree.num_subtree_context_rep_points = num_subtree_context_rep_points
        cfg.subtree.num_candi_rep_points_per_subtree = num_candi_rep_points_per_subtree
        cfg.subtree.num_geometric_properties = num_geometric_properties
        cfg.dataset.dim = all_data_i[0]

    cfg.subtree.max_num_subtrees = max_num_subtrees
    cfg.subtree.num_maps_each_point = 2
    dim = cfg.dataset.dim

    if num_datas > 0:
        cat_points = []
        cat_geometric_properties_wo_padding = []

        all_geometric_properties = []
        all_subtree_context_rep_points = []
        all_subtree_candi_rep_points = []
        all_subtree_scores = []
        all_masks = []

        for all_data_i in all_datas:
            subtree_context_rep_points_i, subtree_candi_rep_points_i, geometric_properties_i, subtree_scores_i, masks_i, cat_points_i, cat_geometric_properties_wo_padding_i = \
                process_subtree_data(cfg, all_data_i)

            cat_points.append(cat_points_i)
            cat_geometric_properties_wo_padding.append(cat_geometric_properties_wo_padding_i)

            all_subtree_context_rep_points.append(subtree_context_rep_points_i)
            all_subtree_candi_rep_points.append(subtree_candi_rep_points_i)
            all_geometric_properties.append(geometric_properties_i)
            all_subtree_scores.append(subtree_scores_i)
            all_masks.append(masks_i)

        cat_points = np.concatenate(cat_points, axis=0)
        cat_geometric_properties_wo_padding = np.concatenate(cat_geometric_properties_wo_padding, axis=0)
        all_subtree_context_rep_points = np.concatenate(all_subtree_context_rep_points, axis=0)
        all_subtree_candi_rep_points = np.concatenate(all_subtree_candi_rep_points, axis=0)
        all_geometric_properties = np.concatenate(all_geometric_properties, axis=0)
        all_subtree_scores = np.concatenate(all_subtree_scores, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
    else:
        all_subtree_context_rep_points, all_subtree_candi_rep_points, all_geometric_properties, all_subtree_scores, all_masks, cat_points, cat_geometric_properties_wo_padding = \
            process_subtree_data(cfg, all_datas[0])

    points_mean, points_std, cat_points, [all_subtree_context_rep_points, all_subtree_candi_rep_points] = normalize_features(
        cat_points, [all_subtree_context_rep_points, all_subtree_candi_rep_points], reshape_train_features=False)
    geometric_properties_mean, geometric_properties_std, cat_geometric_properties_wo_padding, [all_geometric_properties] = normalize_features(
        cat_geometric_properties_wo_padding, [all_geometric_properties], reshape_train_features=False)

    normalization_infos = np.concatenate([points_mean, points_std, geometric_properties_mean, geometric_properties_std])
    assert normalization_infos.shape[0] == 2 * (dim * cfg.subtree.num_maps_each_point + cfg.subtree.num_geometric_properties)

    for subtree_normalization_infos_path in subtree_normalization_infos_paths:
        with open(subtree_normalization_infos_path, 'wb') as f:
            f.write(struct.pack('l', cfg.subtree.max_num_subtrees))
            f.write(struct.pack('l', cfg.subtree.num_subtree_context_rep_points))
            f.write(struct.pack('l', cfg.subtree.num_candi_rep_points_per_subtree))
            f.write(struct.pack('l', cfg.subtree.num_maps_each_point))
            f.write(struct.pack('l', cfg.subtree.num_geometric_properties))
            bytes = normalization_infos.tobytes()
            # print(f'data_type = {data_type}, bytes.len = {len(bytes)}')
            f.write(bytes)

    # print(f'max_num_subtrees: {cfg.subtree.max_num_subtrees}, num_subtree_context_rep_points: {cfg.subtree.num_subtree_context_rep_points}, '
    #       f'num_candi_rep_points_per_subtree: {cfg.subtree.num_candi_rep_points_per_subtree}, num_maps_each_subtree: {cfg.subtree.num_maps_each_point},'
    #       f'num_geometric_properties: {cfg.subtree.num_geometric_properties}')
    # print(f'normalization_infos: {normalization_infos}')


    # score_max = np.max(all_subtree_scores)
    # score_min = np.min(all_subtree_scores)
    # score_mean = np.mean(all_subtree_scores)

    data_size = all_subtree_context_rep_points.shape[0]

    shuffle_idxes = np.arange(0, data_size, dtype=np.int64)
    np.random.shuffle(shuffle_idxes)

    N_train = int(data_size * 0.8)
    train_idxes = shuffle_idxes[0:N_train]
    validation_idxes = shuffle_idxes[N_train:]

    # train_subtree_context_rep_points = all_subtree_context_rep_points[train_idxes]
    # train_subtree_scores = all_subtree_scores[train_idxes]
    # train_subtree_candi_rep_points = all_subtree_candi_rep_points[train_idxes]
    # train_geometric_properties = all_geometric_properties[train_idxes]
    # train_masks = all_masks[train_idxes]

    train_subtree_context_rep_points = all_subtree_context_rep_points
    train_subtree_scores = all_subtree_scores
    train_subtree_candi_rep_points = all_subtree_candi_rep_points
    train_geometric_properties = all_geometric_properties
    train_masks = all_masks

    validation_subtree_context_rep_points = all_subtree_context_rep_points[validation_idxes]
    validation_subtree_scores = all_subtree_scores[validation_idxes]
    validation_subtree_candi_rep_points = all_subtree_candi_rep_points[validation_idxes]
    validation_geometric_properties = all_geometric_properties[validation_idxes]
    validation_masks = all_masks[validation_idxes]

    dtype = np.float32
    if cfg.subtree_model.use_float64:
        dtype = np.float64

    # print(f'train_subtree_context_rep_points.shape: {train_subtree_context_rep_points.shape}')
    # print(f'train_subtree_candi_rep_points.shape: {train_subtree_candi_rep_points.shape}')
    # print(f'train_subtree_scores.shape: {train_subtree_scores.shape}')
    # print(f'train_geometric_properties.shape: {train_geometric_properties.shape}')
    # print(f'validation_subtree_context_rep_points.shape: {validation_subtree_context_rep_points.shape}')
    # print(f'validation_subtree_candi_rep_points.shape: {validation_subtree_candi_rep_points.shape}')
    # print(f'validation_subtree_scores.shape: {validation_subtree_scores.shape}')
    # print(f'validation_geometric_properties.shape: {validation_geometric_properties.shape}')

    train_data = (train_subtree_context_rep_points.astype(dtype), train_subtree_candi_rep_points.astype(dtype),
                  train_geometric_properties.astype(dtype), train_subtree_scores.astype(dtype), train_masks.astype(dtype))
    validation_data = (validation_subtree_context_rep_points.astype(dtype), validation_subtree_candi_rep_points.astype(dtype),
                 validation_geometric_properties.astype(dtype), validation_subtree_scores.astype(dtype), validation_masks.astype(dtype))
    return train_data, validation_data, validation_data


def process_split_data(cfg, data):
    num_candi_rep_points_all_splits = cfg.split.max_num_splits * cfg.split.num_candi_rep_points_per_split
    (dim, num_mbrs, lbds, ubds, _, num_points_from_parent_node,
     num_split_context_rep_points, num_candi_rep_points_per_split,
     num_geometric_properties, data_num_splits,
     all_node_mbr_mins, all_node_mbr_maxs,
     all_split_context_rep_points, all_split_candi_rep_points, all_geometric_properties,
     all_split_scores) = data

    all_split_scores = all_split_scores[:, 0:cfg.split.max_num_splits]
    all_split_candi_rep_points = all_split_candi_rep_points[:,
                                    0:cfg.split.max_num_splits * num_candi_rep_points_per_split, :]
    all_geometric_properties = all_geometric_properties[:, 0:cfg.split.max_num_splits, :]

    lbds_local = batch_repeat(all_node_mbr_mins, num_split_context_rep_points)
    ubds_local = batch_repeat(all_node_mbr_maxs, num_split_context_rep_points)
    diff_local = ubds_local - lbds_local

    lbds_global = lbds
    ubds_global = ubds
    diff_global = ubds_global - lbds_global

    all_split_context_rep_points = scale_and_expand_points(all_split_context_rep_points, lbds_local, diff_local, lbds_global,
                                                      diff_global)

    if num_candi_rep_points_all_splits <= num_split_context_rep_points:
        lbds_local = lbds_local[:, 0:num_candi_rep_points_all_splits, :]
        diff_local = diff_local[:, 0:num_candi_rep_points_all_splits, :]
    else:
        lbds_local = batch_repeat(all_node_mbr_mins, num_candi_rep_points_all_splits)
        ubds_local = batch_repeat(all_node_mbr_maxs, num_candi_rep_points_all_splits)
        diff_local = ubds_local - lbds_local

    all_split_candi_rep_points = scale_and_expand_points(all_split_candi_rep_points, lbds_local, diff_local,
                                                            lbds_global, diff_global)

    split_point_max = np.max(all_split_candi_rep_points)
    split_point_min = np.min(all_split_candi_rep_points)
    split_point_mean = np.mean(all_split_candi_rep_points)

    all_points = [np.reshape(all_split_context_rep_points, (-1, all_split_context_rep_points.shape[-1]))]
    data_size = all_split_context_rep_points.shape[0]
    assert all_split_candi_rep_points.shape[1] == num_candi_rep_points_all_splits

    all_geometric_properties_wo_padding = []
    for i in range(data_size):
        num_splits = data_num_splits[i]
        split_candi_rep_points_i \
            = all_split_candi_rep_points[i, 0:num_splits * num_candi_rep_points_per_split, :]
        assert len(split_candi_rep_points_i.shape) == 2
        assert split_candi_rep_points_i.shape[0] == num_splits * num_candi_rep_points_per_split and \
               split_candi_rep_points_i.shape[1] == dim * cfg.split.num_maps_each_point
        all_points.append(split_candi_rep_points_i)

        geometric_properties_i = all_geometric_properties[i, 0:num_splits, :]
        all_geometric_properties_wo_padding.append(geometric_properties_i)

    all_points = np.concatenate(all_points, axis=0)
    all_geometric_properties_wo_padding = np.concatenate(all_geometric_properties_wo_padding, axis=0)

    data_size = all_split_context_rep_points.shape[0]
    all_masks = np.zeros(shape=[data_size, cfg.split.max_num_splits], dtype=all_split_scores.dtype)
    for i in range(data_size):
        num_splits = data_num_splits[i]
        all_masks[i, 0:num_splits] = 1

    return all_split_context_rep_points, all_split_candi_rep_points, all_geometric_properties, all_split_scores, all_masks, all_points, all_geometric_properties_wo_padding

def load_split_data(cfg):
    split_records_paths, split_normalization_infos_paths, _, _ = config.get_data_dirs(cfg)

    num_datas = len(split_records_paths)
    all_datas = []
    max_num_splits = 0

    for split_records_path in split_records_paths:
        all_data_i = data_utils.load_split_records(split_records_path)
        all_datas.append(all_data_i)
        (dim, num_mbrs, lbds, ubds, max_num_splits_i, num_points_from_parent_node,
         num_split_context_rep_points, num_candi_rep_points_per_split,
         num_geometric_properties, data_num_splits,
         all_node_mbr_mins, all_node_mbr_maxs,
         all_split_context_rep_points, all_split_candi_rep_points, all_geometric_properties,
         all_split_scores) = all_data_i

        if max_num_splits_i > max_num_splits:
            max_num_splits = max_num_splits_i

        cfg.split.num_points_from_parent_node = num_points_from_parent_node
        cfg.split.num_split_context_rep_points = num_split_context_rep_points
        cfg.split.num_candi_rep_points_per_split = num_candi_rep_points_per_split
        cfg.split.num_geometric_properties = num_geometric_properties
        cfg.dataset.dim = all_data_i[0]

    cfg.split.max_num_splits = max_num_splits
    cfg.split.num_maps_each_point = 2
    dim = cfg.dataset.dim


    if num_datas > 0:
        cat_points = []
        cat_geometric_properties_wo_padding = []

        all_geometric_properties = []
        all_split_context_rep_points = []
        all_split_candi_rep_points = []
        all_split_scores = []
        all_masks = []

        for all_data_i in all_datas:
            split_context_rep_points_i, split_candi_rep_points_i, geometric_properties_i, split_scores_i, masks_i, cat_points_i, cat_geometric_properties_wo_padding_i =\
                process_split_data(cfg, all_data_i)

            cat_points.append(cat_points_i)
            cat_geometric_properties_wo_padding.append(cat_geometric_properties_wo_padding_i)

            all_split_context_rep_points.append(split_context_rep_points_i)
            all_split_candi_rep_points.append(split_candi_rep_points_i)
            all_geometric_properties.append(geometric_properties_i)
            all_split_scores.append(split_scores_i)
            all_masks.append(masks_i)

        cat_points = np.concatenate(cat_points, axis=0)
        cat_geometric_properties_wo_padding = np.concatenate(cat_geometric_properties_wo_padding, axis=0)
        all_split_context_rep_points = np.concatenate(all_split_context_rep_points, axis=0)
        all_split_candi_rep_points = np.concatenate(all_split_candi_rep_points, axis=0)
        all_geometric_properties = np.concatenate(all_geometric_properties, axis=0)
        all_split_scores = np.concatenate(all_split_scores, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
    else:
        all_split_context_rep_points, all_split_candi_rep_points, all_geometric_properties, all_split_scores, all_masks, cat_points, cat_geometric_properties_wo_padding =\
            process_split_data(cfg, all_datas[0])


    # all_points = np.concatenate((all_split_context_rep_points, all_split_candi_rep_points), axis=1)
    points_mean, points_std, cat_points, [all_split_context_rep_points, all_split_candi_rep_points] = normalize_features(
        cat_points, [all_split_context_rep_points, all_split_candi_rep_points], reshape_train_features=False)
    geometric_properties_mean, geometric_properties_std, all_geometric_properties_wo_padding, [all_geometric_properties] = normalize_features(
        cat_geometric_properties_wo_padding, [all_geometric_properties], reshape_train_features=False)


    normalization_infos = np.concatenate([points_mean, points_std, geometric_properties_mean, geometric_properties_std])
    assert normalization_infos.shape[0] == 2 * (dim * cfg.split.num_maps_each_point + cfg.split.num_geometric_properties)

    for split_normalization_infos_path in split_normalization_infos_paths:
        with open(split_normalization_infos_path, 'wb') as f:
            f.write(struct.pack('l', cfg.split.max_num_splits))
            f.write(struct.pack('l', cfg.split.num_split_context_rep_points))
            f.write(struct.pack('l', cfg.split.num_candi_rep_points_per_split))
            f.write(struct.pack('l', cfg.split.num_maps_each_point))
            f.write(struct.pack('l', cfg.split.num_geometric_properties))
            bytes = normalization_infos.tobytes()
            # print(f'data_type = {data_type}, bytes.len = {len(bytes)}')
            f.write(bytes)

    # print(f'max_num_splits: {cfg.split.max_num_splits}, num_split_context_rep_points: {cfg.split.num_split_context_rep_points}, '
    #       f'num_candi_rep_points_per_split: {cfg.split.num_candi_rep_points_per_split}, num_maps_each_split: {cfg.split.num_maps_each_point},'
    #       f'num_geometric_properties: {cfg.split.num_geometric_properties}')
    # print(f'normalization_infos: {normalization_infos}')


    # score_max = np.max(all_split_scores)
    # score_min = np.min(all_split_scores)
    # score_mean = np.mean(all_split_scores)

    data_size = all_split_context_rep_points.shape[0]
    shuffle_idxes = np.arange(0, data_size, dtype=np.int64)
    np.random.shuffle(shuffle_idxes)

    N_train = int(data_size * 0.8)
    train_idxes = shuffle_idxes[0:N_train]
    validation_idxes = shuffle_idxes[N_train:]

    # train_split_context_rep_points = all_split_context_rep_points[train_idxes]
    # train_split_scores = all_split_scores[train_idxes]
    # train_split_candi_rep_points = all_split_candi_rep_points[train_idxes]
    # train_geometric_properties = all_geometric_properties[train_idxes]
    # train_masks = all_masks[train_idxes]

    train_split_context_rep_points = all_split_context_rep_points
    train_split_scores = all_split_scores
    train_split_candi_rep_points = all_split_candi_rep_points
    train_geometric_properties = all_geometric_properties
    train_masks = all_masks

    validation_split_context_rep_points = all_split_context_rep_points[validation_idxes]
    validation_split_scores = all_split_scores[validation_idxes]
    validation_split_candi_rep_points = all_split_candi_rep_points[validation_idxes]
    validation_geometric_properties = all_geometric_properties[validation_idxes]
    validation_masks = all_masks[validation_idxes]

    dtype = np.float32
    if cfg.split_model.use_float64:
        dtype = np.float64

    # print(f'train_split_context_rep_points.shape: {train_split_context_rep_points.shape}')
    # print(f'train_split_candi_rep_points.shape: {train_split_candi_rep_points.shape}')
    # print(f'train_split_scores.shape: {train_split_scores.shape}')
    # print(f'train_geometric_properties.shape: {train_geometric_properties.shape}')
    # print(f'validation_split_context_rep_points.shape: {validation_split_context_rep_points.shape}')
    # print(f'validation_split_candi_rep_points.shape: {validation_split_candi_rep_points.shape}')
    # print(f'validation_split_scores.shape: {validation_split_scores.shape}')
    # print(f'validation_geometric_properties.shape: {validation_geometric_properties.shape}')

    train_data = (train_split_context_rep_points.astype(dtype), train_split_candi_rep_points.astype(dtype),
                  train_geometric_properties.astype(dtype), train_split_scores.astype(dtype), train_masks.astype(dtype))
    validation_data = (validation_split_context_rep_points.astype(dtype), validation_split_candi_rep_points.astype(dtype),
                 validation_geometric_properties.astype(dtype), validation_split_scores.astype(dtype), validation_masks.astype(dtype))
    return train_data, validation_data


if __name__ == '__main__':
    cfg = config.getConfigs()
    load_subtree_data(cfg)
    # load_split_data(cfg)