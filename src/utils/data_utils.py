from . import file_utils
import numpy as np
import struct
import math



def load_split_records(path):
    with open(path, 'rb') as f:
        file_contents = f.read()

    curr = 0
    data_isLeaf = []
    data_num_splits = []

    all_node_mbr_mins = []
    all_node_mbr_maxs = []

    all_split_context_rep_points = []
    all_split_scores = []
    all_split_mbrs = []
    all_split_candi_rep_points = []
    all_geometric_properties = []

    dim = struct.unpack('i', file_contents[curr:curr + 4])[0]
    num_mbrs = struct.unpack('i', file_contents[curr + 4:curr + 8])[0]
    num_splits_threshold = struct.unpack('i', file_contents[curr + 8:curr + 12])[0]
    num_geometric_properties = struct.unpack('i', file_contents[curr + 12:curr + 16])[0]
    curr += 16

    l = curr + 8 * dim
    lbds = struct.unpack('d' * dim, file_contents[curr:l])
    lbds = np.array(lbds, dtype=np.float64)
    curr = l
    l = curr + 8 * dim
    ubds = struct.unpack('d' * dim, file_contents[curr:l])
    ubds = np.array(ubds, dtype=np.float64)

    curr = l
    max_num_splits = 0

    num_split_context_rep_points = int(math.pow(2, dim) + 1 + 0.5)
    if dim < 4:
        num_split_context_rep_points = int(math.pow(3, dim) + 0.5)
    num_points_from_parent_node = num_split_context_rep_points
    num_split_context_rep_points += num_mbrs

    # print(
    #     f'dim = {dim}, num_mbrs = {num_mbrs}, num_splits_threshold = {num_splits_threshold}, num_geometric_properties = {num_geometric_properties}')
    # print(f'lbds = {lbds}, ubds = {ubds}')
    # print(f'num_split_context_rep_points = {num_split_context_rep_points}')

    num_candi_rep_points_per_split = 2 * int(math.pow(2, dim) + 0.5)
    while curr < len(file_contents):
        isLeafInt = struct.unpack('i', file_contents[curr:curr+4])[0]
        num_splits = struct.unpack('i', file_contents[curr+4:curr+8])[0]
        curr += 8

        data_isLeaf.append(isLeafInt)
        data_num_splits.append(num_splits)
        if num_splits > max_num_splits:
            max_num_splits = num_splits
        if num_splits > num_splits_threshold:
            print(f'num_splits = {num_splits}')
        assert num_splits <= num_splits_threshold

        l = curr + 8 * dim
        m_pLow = struct.unpack('d' * dim, file_contents[curr:l])
        m_pLow = np.array(m_pLow, dtype=np.float64)
        curr = l
        l = curr + 8 * dim
        m_pHigh = struct.unpack('d' * dim, file_contents[curr:l])
        m_pHigh = np.array(m_pHigh, dtype=np.float64)
        # diff = max_coords - min_coords
        # min_diff = np.min(diff)
        all_node_mbr_mins.append(m_pLow)
        all_node_mbr_maxs.append(m_pHigh)

        curr = l
        l = curr + 8 * dim * num_split_context_rep_points
        split_context_rep_points = struct.unpack('d' * dim * num_split_context_rep_points, file_contents[curr:l])
        split_context_rep_points = np.array(split_context_rep_points, dtype=np.float64)
        split_context_rep_points = np.reshape(split_context_rep_points, newshape=(num_split_context_rep_points, dim))
        all_split_context_rep_points.append(split_context_rep_points)

        curr = l
        l = curr + 8 * num_splits * dim * num_candi_rep_points_per_split
        split_candi_rep_points = struct.unpack('d' * num_splits * dim * num_candi_rep_points_per_split, file_contents[curr:l])
        split_candi_rep_points = np.array(split_candi_rep_points, dtype=np.float64)
        split_candi_rep_points = np.reshape(split_candi_rep_points, newshape=(num_splits * num_candi_rep_points_per_split, dim))

        curr = l
        l = curr + 8 * dim * num_candi_rep_points_per_split
        padding_point = struct.unpack('d' * dim * num_candi_rep_points_per_split, file_contents[curr:l])
        padding_point = np.array(padding_point, dtype=np.float64)
        padding_point = np.reshape(padding_point, newshape=(1, num_candi_rep_points_per_split, dim))
        if num_splits < num_splits_threshold:
            padding_data = np.repeat(padding_point, num_splits_threshold - num_splits, axis=0)
            padding_data = np.reshape(padding_data, newshape=(-1, padding_data.shape[-1]))
            split_candi_rep_points = np.concatenate([split_candi_rep_points, padding_data], axis=0)
        all_split_candi_rep_points.append(split_candi_rep_points)

        curr = l
        l = curr + 8 * num_splits * num_geometric_properties
        geometric_properties = struct.unpack('d' * num_splits * num_geometric_properties, file_contents[curr:l])
        geometric_properties = np.array(geometric_properties, dtype=np.float64)
        geometric_properties = np.reshape(geometric_properties, newshape=(num_splits, num_geometric_properties))
        curr = l
        l = curr + 8 * num_geometric_properties
        padding_geometric_properties = struct.unpack('d' * num_geometric_properties, file_contents[curr:l])
        padding_geometric_properties = np.array(padding_geometric_properties, dtype=np.float64)
        if num_splits < num_splits_threshold:
            padding_data = np.reshape(padding_geometric_properties, newshape=(1, -1))
            padding_data = np.repeat(padding_data, (num_splits_threshold - num_splits), axis=0)
            geometric_properties = np.concatenate([geometric_properties, padding_data], axis=0)
        all_geometric_properties.append(geometric_properties)

        curr = l
        l = curr + 8 * num_splits
        split_scores = struct.unpack('d' * num_splits, file_contents[curr:l])
        split_scores = np.array(split_scores, dtype=np.float64).tolist()
        curr = l
        l = curr + 8
        padding_score = struct.unpack('d', file_contents[curr:l])[0]
        if num_splits < num_splits_threshold:
            padding_scores = [padding_score] * (num_splits_threshold - num_splits)
            split_scores.extend(padding_scores)
        all_split_scores.append(split_scores)

        curr = l

    data_num_splits = np.array(data_num_splits, dtype=np.int64)

    all_node_mbr_mins = np.array(all_node_mbr_mins, dtype=np.float64)
    all_node_mbr_maxs = np.array(all_node_mbr_maxs, dtype=np.float64)

    all_split_context_rep_points = np.array(all_split_context_rep_points, dtype=np.float64)
    all_split_scores = np.array(all_split_scores, dtype=np.float64)
    all_split_candi_rep_points = np.array(all_split_candi_rep_points, dtype=np.float64)
    all_geometric_properties = np.array(all_geometric_properties, dtype=np.float64)


    return (dim, num_mbrs, lbds, ubds, max_num_splits, num_points_from_parent_node,
            num_split_context_rep_points, num_candi_rep_points_per_split,
            num_geometric_properties, data_num_splits,
            all_node_mbr_mins, all_node_mbr_maxs,
            all_split_context_rep_points, all_split_candi_rep_points, all_geometric_properties,
            all_split_scores,)


def load_subtree_records(path):
    with open(path, 'rb') as f:
        file_contents = f.read()

    curr = 0
    data_num_subtrees = []

    all_parent_mbr_mins = []
    all_parent_mbr_maxs = []

    all_subtree_context_rep_points = []
    all_subtree_scores = []
    all_subtree_candi_rep_points = []
    all_geometric_properties = []

    dim = struct.unpack('i', file_contents[curr:curr + 4])[0]
    num_subtrees_threshold = struct.unpack('i', file_contents[curr + 4:curr + 8])[0]
    num_geometric_properties = struct.unpack('i', file_contents[curr + 8:curr + 12])[0]
    curr += 12

    l = curr + 8 * dim
    lbds = struct.unpack('d' * dim, file_contents[curr:l])
    lbds = np.array(lbds, dtype=np.float64)
    curr = l
    l = curr + 8 * dim
    ubds = struct.unpack('d' * dim, file_contents[curr:l])
    ubds = np.array(ubds, dtype=np.float64)

    curr = l
    max_num_subtrees = 0

    num_subtree_context_rep_points = int(math.pow(2, dim) + 0.5)
    num_candi_rep_points_per_subtree = int(math.pow(2, dim) + 0.5)

    # print(
    #     f'dim = {dim}, num_subtrees_threshold = {num_subtrees_threshold}, num_geometric_properties = {num_geometric_properties}')
    # print(f'lbds = {lbds}, ubds = {ubds}')

    while curr < len(file_contents):
        num_subtrees = struct.unpack('i', file_contents[curr:curr+4])[0]
        curr += 4
        # print(f'num_subtrees = {num_subtrees}')

        data_num_subtrees.append(num_subtrees)
        if num_subtrees > max_num_subtrees:
            max_num_subtrees = num_subtrees
        if num_subtrees > num_subtrees_threshold:
            print(f'num_subtrees = {num_subtrees}')
        assert num_subtrees <= num_subtrees_threshold

        l = curr + 8 * dim
        m_pLow = struct.unpack('d' * dim, file_contents[curr:l])
        m_pLow = np.array(m_pLow, dtype=np.float64)
        curr = l
        l = curr + 8 * dim
        m_pHigh = struct.unpack('d' * dim, file_contents[curr:l])
        m_pHigh = np.array(m_pHigh, dtype=np.float64)
        all_parent_mbr_mins.append(m_pLow)
        all_parent_mbr_maxs.append(m_pHigh)

        curr = l
        l = curr + 8 * dim * num_subtree_context_rep_points
        subtree_context_rep_points = struct.unpack('d' * dim * num_subtree_context_rep_points, file_contents[curr:l])
        subtree_context_rep_points = np.array(subtree_context_rep_points, dtype=np.float64)
        subtree_context_rep_points = np.reshape(subtree_context_rep_points, newshape=(num_subtree_context_rep_points, dim))
        all_subtree_context_rep_points.append(subtree_context_rep_points)

        curr = l
        l = curr + 8 * num_subtrees * dim * num_candi_rep_points_per_subtree
        subtree_candi_rep_points = struct.unpack('d' * num_subtrees * dim * num_candi_rep_points_per_subtree, file_contents[curr:l])
        subtree_candi_rep_points = np.array(subtree_candi_rep_points, dtype=np.float64)
        subtree_candi_rep_points = np.reshape(subtree_candi_rep_points, newshape=(num_subtrees * num_candi_rep_points_per_subtree, dim))

        curr = l
        l = curr + 8 * dim * num_candi_rep_points_per_subtree
        padding_point = struct.unpack('d' * dim * num_candi_rep_points_per_subtree, file_contents[curr:l])
        padding_point = np.array(padding_point, dtype=np.float64)
        padding_point = np.reshape(padding_point, newshape=(1, num_candi_rep_points_per_subtree, dim))
        if num_subtrees < num_subtrees_threshold:
            padding_data = np.repeat(padding_point, num_subtrees_threshold - num_subtrees, axis=0)
            padding_data = np.reshape(padding_data, newshape=(-1, padding_data.shape[-1]))
            subtree_candi_rep_points = np.concatenate([subtree_candi_rep_points, padding_data], axis=0)
        all_subtree_candi_rep_points.append(subtree_candi_rep_points)


        curr = l
        l = curr + 8 * num_subtrees * num_geometric_properties
        geometric_properties = struct.unpack('d' * num_subtrees * num_geometric_properties, file_contents[curr:l])
        geometric_properties = np.array(geometric_properties, dtype=np.float64)
        geometric_properties = np.reshape(geometric_properties, newshape=(num_subtrees, num_geometric_properties))
        # print(f'geometric_properties[0,0] = {geometric_properties[0, 0]}')
        curr = l
        l = curr + 8 * num_geometric_properties
        padding_geometric_properties = struct.unpack('d' * num_geometric_properties, file_contents[curr:l])
        padding_geometric_properties = np.array(padding_geometric_properties, dtype=np.float64)
        if num_subtrees < num_subtrees_threshold:
            padding_data = np.reshape(padding_geometric_properties, newshape=(1, -1))
            padding_data = np.repeat(padding_data, (num_subtrees_threshold - num_subtrees), axis=0)
            geometric_properties = np.concatenate([geometric_properties, padding_data], axis=0)
            # print(f'geometric_feature_padding_data[0,0] = {padding_data[0, 0]}')
        all_geometric_properties.append(geometric_properties)

        curr = l
        l = curr + 8 * num_subtrees
        subtree_scores = struct.unpack('d' * num_subtrees, file_contents[curr:l])
        subtree_scores = np.array(subtree_scores, dtype=np.float64).tolist()

        curr = l
        l = curr + 8
        padding_score = struct.unpack('d', file_contents[curr:l])[0]
        if num_subtrees < num_subtrees_threshold:
            padding_scores = [padding_score] * (num_subtrees_threshold - num_subtrees)
            subtree_scores.extend(padding_scores)
        all_subtree_scores.append(subtree_scores)

        curr = l

    # assert len(best_splits) > 0
    
    data_num_subtrees = np.array(data_num_subtrees, dtype=np.int64)

    all_parent_mbr_mins = np.array(all_parent_mbr_mins, dtype=np.float64)
    all_parent_mbr_maxs = np.array(all_parent_mbr_maxs, dtype=np.float64)

    all_subtree_context_rep_points = np.array(all_subtree_context_rep_points, dtype=np.float64)
    all_subtree_scores = np.array(all_subtree_scores, dtype=np.float64)
    all_subtree_candi_rep_points = np.array(all_subtree_candi_rep_points, dtype=np.float64)
    all_geometric_properties = np.array(all_geometric_properties, dtype=np.float64)

    return (dim, lbds, ubds, max_num_subtrees, 0,
            num_subtree_context_rep_points, num_candi_rep_points_per_subtree,
            num_geometric_properties, data_num_subtrees,
            all_parent_mbr_mins, all_parent_mbr_maxs,
            all_subtree_context_rep_points, all_subtree_candi_rep_points, all_geometric_properties,
            all_subtree_scores)


if __name__ == '__main__':
    pass
