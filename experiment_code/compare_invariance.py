import os
import sys

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sklearn.tree import _tree
from utils import gini, gini_normalized
import numpy as np

def calc_gini(x, y, th):
    mask_left = x > th
    left_y, right_y = y[mask_left], y[~mask_left]
    scale_left, scale_right = len(left_y) / len(y), len(right_y) / len(y)
    assert scale_left + scale_right == 1, f'{scale_left + scale_right} not equal 1'
    return (scale_left * gini_normalized(sum(left_y > 0.5), len(left_y)) +
            scale_right * gini_normalized(sum(right_y > 0.5), len(right_y)))

def find_optimal_cut(x, y):
    n = len(y)
    assert n > 1, "n must be greater than 1"
    sorted_index_arr = np.argsort(x)

    total, pos = len(y), sum(y > 0.5)
    pos_cum, total_cum, th = 0, 0, 0
    best_gini = np.inf
    for i in range(n - 1):
        curr_index = sorted_index_arr[i]
        total_cum += 1
        pos_cum += y[curr_index] > 0.5
        # if possible to split in between
        if x[sorted_index_arr[i + 1]] > x[sorted_index_arr[i]]:
            left_pos, left_total, right_pos, right_total = pos_cum, total_cum, pos - pos_cum, total - total_cum
            scale_left, scale_right = left_total / total, right_total / total
            assert scale_left + scale_right == 1, f'{scale_left + scale_right} not equal 1'
            gini_tmp = (scale_left * gini_normalized(left_pos, left_total) +
                        scale_right * gini_normalized(right_pos, right_total))
            if gini_tmp < best_gini:
                best_gini = gini_tmp
                th = (x[sorted_index_arr[i+1]] + x[sorted_index_arr[i]]) / 2
    # if best_gini == np.inf: best_gini = -1
    return best_gini, th


def compare_inv(clf, dataset):
    tree_ = clf.tree_
    all_regrets, all_depths, all_ind_len = [], [], []

    for i, env in enumerate(dataset):
        currX, currY = env.X, env.y
        curr_regrets, curr_depths, curr_ind_len = [], [], []

        def recurse(node_id, depth, indecies):
            if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
                feature_id = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]

                x = currX[indecies, feature_id]
                y = currY[indecies]
                mask_left = x > threshold
                left_indices, right_indices = indecies[mask_left], indecies[~mask_left]
                if len(indecies) <= 1:
                    regret = -1
                    # print(f'len(indecies): {len(indecies)}, len(left_indices): {len(left_indices)}, len(right_indices): {len(right_indices)}')
                elif (x == x[0]).all():
                    regret = -1
                    # print(f'gini: {calc_gini(x, y, threshold)}, {gini_normalized(sum(y), len(y))} no split found, {x}')
                else:
                    gini = calc_gini(x, y, threshold)
                    optimal_gini, optimal_th = find_optimal_cut(x, y)
                    regret = gini - optimal_gini
                    assert regret >= -0.1

                curr_regrets.append(regret)
                curr_depths.append(depth)
                curr_ind_len.append(len(indecies))

                recurse(tree_.children_left[node_id], depth + 1, left_indices)
                recurse(tree_.children_right[node_id], depth + 1, right_indices)
            else:
                curr_regrets.append(-1)
                curr_depths.append(depth)
                curr_ind_len.append(len(indecies))

        recurse(0, 1, np.arange(len(currY)))
        all_regrets.append(curr_regrets)
        all_depths.append(curr_depths)
        all_ind_len.append(curr_ind_len)

    # Print regrets
    depth_arr = np.array(all_depths[0])
    all_gini_means = []
    for env_num in range(len(dataset)):
        curr_env_means = []
        curr_env_regrets = np.array(all_regrets[env_num])
        for depth in sorted(set(depth_arr)):
            curr_depth_indices = np.where(depth_arr == depth)
            curr_depth_regrets = curr_env_regrets[curr_depth_indices]
            relevant_regrets = curr_depth_regrets[np.where(curr_depth_regrets != -1)]
            curr_env_means.append(np.mean(relevant_regrets))
            if not np.mean(relevant_regrets) or np.mean(relevant_regrets) == 0:
                print(f'depth: {depth}, env: {env_num}, mean: {np.mean(relevant_regrets)}, {curr_depth_regrets}')

        all_gini_means.append(curr_env_means)
        print([round(elem * 100, 2) for elem in curr_env_means])
    print('finished...')
