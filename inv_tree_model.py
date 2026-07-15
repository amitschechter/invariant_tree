import numpy as np


class Node:
    def __init__(self, lambda_val, debug=False):
        self.dim = 0
        self.leaf = True
        self.th, self.prob = 0, 0.5
        self.projW = None
        self.left, self.right = None, None
        self.debug = debug
        self.lambda_val = lambda_val #10000.0 # lambda_val: weight for regret in loss calculation

    def gini(self, npos, n):
        p = npos / n if n > 0 else 0
        gini_score = 2 * p * (1 - p)
        return gini_score

    def populate_individual_envs(self, x_arr, y_arr, w_arr, n_envs, threshold, th_index,
                                 left_pos, left_totals, right_pos, right_totals, all_env_ginis):
        # save individual environment counts
        for env_num in range(n_envs):

            env_left_mask = x_arr[env_num] <= threshold

            curr_w_left, curr_y_left = w_arr[env_num][env_left_mask], y_arr[env_num][env_left_mask]
            left_pos[env_num][th_index] = np.sum(curr_w_left * curr_y_left)
            left_totals[env_num][th_index] = np.sum(curr_w_left)

            curr_w_right, curr_y_right = w_arr[env_num][~env_left_mask], y_arr[env_num][~env_left_mask]
            right_pos[env_num][th_index] = np.sum(curr_w_right * curr_y_right)
            right_totals[env_num][th_index] = np.sum(curr_w_right)

            n_total_env = left_totals[env_num][th_index] + right_totals[env_num][th_index]
            if n_total_env <= 0:
                all_env_ginis[env_num][th_index] = 0
            else:
                weight_left = left_totals[env_num][th_index] / n_total_env
                weight_right = right_totals[env_num][th_index] / n_total_env

                all_env_ginis[env_num][th_index] = (
                        weight_left * self.gini(left_pos[env_num][th_index], left_totals[env_num][th_index]) +
                        weight_right * self.gini(right_pos[env_num][th_index], right_totals[env_num][th_index]))

    def get_max_regret(self, all_env_ginis, left_pos, right_pos, left_totals, right_totals,
                       n_thresholds, n_envs):
        left_pos_sum = np.sum(left_pos, axis=0).reshape(1, n_thresholds)
        right_pos_sum = np.sum(right_pos, axis=0).reshape(1, n_thresholds)
        left_total_sum = np.sum(left_totals, axis=0).reshape(1, n_thresholds)
        right_total_sum = np.sum(right_totals, axis=0).reshape(1, n_thresholds)

        left_pos_excluding_e = left_pos_sum - left_pos  # size n_envs X n_thresholds
        left_total_excluding_e = left_total_sum - left_totals
        right_pos_excluding_e = right_pos_sum - right_pos
        right_total_excluding_e = right_total_sum - right_totals

        def gini_from_arrays_all_but_e(env_num, th_ind):
            left_pos, left_total = left_pos_excluding_e[env_num][th_ind], left_total_excluding_e[env_num][th_ind]
            right_pos, right_total = right_pos_excluding_e[env_num][th_ind], right_total_excluding_e[env_num][th_ind]
            n_total = left_total_excluding_e[env_num][th_ind] + right_total_excluding_e[env_num][th_ind]
            if n_total <= 0:
                return 0
            left_gini, right_gini = self.gini(left_pos, left_total), self.gini(right_pos, right_total)
            return (left_total / n_total) * left_gini + (right_total / n_total) * right_gini

        gini_without_e_mat = np.empty((n_envs, n_thresholds))
        for env_num in range(n_envs):
            gini_without_e_mat[env_num, :] = np.array(
                [gini_from_arrays_all_but_e(env_num, th_ind) for th_ind in range(n_thresholds)])

        # optimum for single env
        min_t_e_gini_arr = np.min(all_env_ginis, axis=1)  # array of size n_envs, with the opt gini for each env

        # optimum without env
        min_t_without_e = np.argmin(gini_without_e_mat, axis=1)  # list of size n_envs, with index of best threshold
        min_t_without_e_gini_arr = all_env_ginis[
            np.arange(len(all_env_ginis)), min_t_without_e]  # arr of size n_envs

        # regret for each env is the difference between optimum without the env and absolute optimum
        regrets = min_t_without_e_gini_arr - min_t_e_gini_arr  # arr of size n_envs
        assert (regrets >= 0).all()
        return np.max(regrets), np.sum(regrets)

    def get_feature_opt(self, x_arr, y_arr, w_arr):
        """
        :param x_arr: an array of size n_envs, where each element is a list of size n_datapoints
        :param y_arr: an array of size n_envs, where each element is a list of size n_datapoints
        :param w_arr: an array of size n_envs, where each element is a list of size n_datapoints
        :return: the optimal loss and the corresponding threshold
        """
        n_envs = len(y_arr)
        all_x_vals = sorted(list(set(np.concatenate(x_arr))))
        all_thresholds = np.convolve(all_x_vals, np.ones(2), 'valid') / 2  # moving average
        n_thresholds = len(all_thresholds)

        # Should be leaf
        if len(all_thresholds) < 1 or len(all_x_vals) <= 1:
            return np.inf, 0

        # initialize
        total_ginis = np.empty_like(all_thresholds)  # size n_thresholds of ginis for the whole dataset (flattened)
        left_pos, left_totals = np.empty((n_envs, n_thresholds)), np.empty((n_envs, n_thresholds))
        right_pos, right_totals = np.empty((n_envs, n_thresholds)), np.empty((n_envs, n_thresholds))
        all_env_ginis = np.empty((n_envs, n_thresholds))

        flattened_y, flattened_x = np.concatenate(y_arr), np.concatenate(x_arr)  # n_datapoints, n_datapoints X dim
        flattened_w = np.concatenate(w_arr)
        n_pos_total, n_total = np.sum(flattened_w * flattened_y), len(flattened_y)

        for th_index, threshold in enumerate(all_thresholds):
            # calculate gini for the whole (flatten) dataset
            left_mask = flattened_x <= threshold
            n_total_left = np.sum(flattened_w[left_mask])
            n_total_right = n_total - n_total_left
            n_pos_left = np.sum(flattened_w[left_mask] * flattened_y[left_mask])
            n_pos_right = n_pos_total - n_pos_left
            weight_left, weight_right = n_total_left / n_total, n_total_right / n_total
            gini_tmp = (weight_left * self.gini(n_pos_left, n_total_left) +
                        weight_right * self.gini(n_pos_right, n_total_right))
            total_ginis[th_index] = gini_tmp

            self.populate_individual_envs(x_arr, y_arr, w_arr, n_envs, threshold, th_index,
                                          left_pos, left_totals, right_pos, right_totals, all_env_ginis)

        max_regret, total_regret = self.get_max_regret(all_env_ginis, left_pos, right_pos, left_totals, right_totals,
                                         n_thresholds, n_envs)

        # optimum for combined data
        min_t_ind = np.argmin(total_ginis)
        opt_total_gini = total_ginis[min_t_ind]

        loss = opt_total_gini + self.lambda_val * max_regret#total_regret #max_regret

        return loss, all_thresholds[min_t_ind]

    '''
    Dataset:
        x: list of size n_envs, where every element is a list
        y: list of n_envs binary integers
        w: list of n_envs, where every element is a list. All elems are 1 for now
        n_features: int
        n_envs: int
    '''

    def build(self, dataset, max_depth, n_proj, index_arr):
        """
        :param dataset: as detailed above
        :param max_depth: max_depth from this node down
        :param n_proj: number of projections to consider
        :param index_arr: an array of size n_envs, where each element is an index array for the nev
        """
        dim, n_envs = dataset.n_features, dataset.n_envs
        if index_arr is None:
            index_arr = [np.arange(len(env_y)) for env_y in dataset.y]
        n_array = [len(env_index) for env_index in index_arr]
        total_n = sum(n_array)

        x = [env_x[index_arr[env_num]] for env_num, env_x in enumerate(dataset.x)]
        y = [env_y[index_arr[env_num]] for env_num, env_y in enumerate(dataset.y)]
        w = [env_w[index_arr[env_num]] for env_num, env_w in enumerate(dataset.w)]

        # prediction at the node is a weighted average, using the weight vector w
        wtot = np.sum(np.sum(env_w) for env_w in w)
        self.prob = np.sum(np.sum(w[env_num] * y[env_num]) for env_num in range(n_envs))
        self.prob /= wtot

        # if len(all_y_set) == 1, then the node is uniform - leaf
        flatten_y = np.concatenate(y)
        all_y_set = set(flatten_y)
        if self.debug:
            self.n_samples = total_n
            flatten_y = np.concatenate(y)
            self.values = [len(flatten_y) - np.sum(flatten_y), np.sum(flatten_y)]
            self.gini_score = self.gini(np.sum(flatten_y), len(flatten_y))

        # node is a leaf
        if max_depth <= 0 or total_n <= 1 or len(all_y_set) <= 1:
            return

        gains, thresholds = [], []
        # Z = np.random.normal(size=(n_proj, dim))
        Z = np.identity(dim)
        for k, curr_proj in enumerate(Z):
            # x_array is a list of n_envs, where each element is a list of size n_datapoints for the envs
            x_array = [np.sum(curr_proj.reshape(1, -1) * x[env_num], axis=1) for env_num in range(n_envs)]
            gain_tmp, th_tmp = self.get_feature_opt(x_array, y, w)
            gains.append(gain_tmp)
            thresholds.append(th_tmp)

        proj_ind = np.argmin(gains)
        self.projW = Z[proj_ind]
        self.th = thresholds[proj_ind]
        self.leaf = False

        x_proj = [np.sum(self.projW * x[env_num], axis=1) for env_num in range(n_envs)]
        mask_left = [curr_x_array <= self.th for curr_x_array in x_proj]
        sel_left = [env_index[mask_left[env_num]] for env_num, env_index in enumerate(index_arr)]
        sel_right = [env_index[~mask_left[env_num]] for env_num, env_index in enumerate(index_arr)]
        total_n_left = sum(len(curr_left) for curr_left in sel_left)
        total_n_right = sum(len(curr_right) for curr_right in sel_right)

        if total_n_left == 0 or total_n_right == 0:
            print(f'total_n_left: {total_n_left}, total_n_right: {total_n_right}')
            print(f'sel_left: {sel_left}, sel_right: {sel_right}')
            print(f'{x_proj}, th: {self.th}')
            print(f'proj_ind: {proj_ind}, thresholds: {thresholds}')
        assert total_n_left > 0 and total_n_right > 0  # if not, should be a leaf
        assert total_n_left + total_n_right == total_n

        self.left = Node(self.lambda_val, self.debug)
        self.left.build(dataset, max_depth - 1, n_proj, sel_left)
        self.right = Node(self.lambda_val, self.debug)
        self.right.build(dataset, max_depth - 1, n_proj, sel_right)

    def print_info(self, depth, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces
        const = int(depth * width ** 1.5)
        spaces = "-" * const

        if not self.leaf:
            print(f"|{spaces} Split rule: x{np.where(self.projW != 0)[0]} <= {self.th}")
            print(f"{' ' * const}   | gini: {self.gini_score}")
        if self.leaf:
            print(f"{' ' * const}   | Predicted prob: {self.prob}")
        print(f"{' ' * const}   | samples: {self.n_samples}")
        print(f"{' ' * const}   | values: {self.values}")

    def print_tree(self, depth):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info(depth)
        if self.left is not None:
            self.left.print_tree(depth + 1)
        if self.right is not None:
            self.right.print_tree(depth + 1)

    def predict(self, x_arr):  # x_arr is a single datapoint
        if self.leaf:
            return self.prob
        x_val = np.sum(np.array(x_arr) * self.projW)
        if x_val <= self.th:
            return self.left.predict(x_arr)
        else:
            return self.right.predict(x_arr)

    def eval(self, dataset):
        """
        :param dataset: includes x and y fields, where each is a list of n_envs items.
                        each env has a list of n_datapoints
        :return: accuracy score for the dataset
        """
        acc, total_n = 0, 0
        for env_num, curr_env_x in enumerate(dataset.x):
            for datapoint_num, curr_x in enumerate(curr_env_x):
                curr_prob = self.predict(curr_x)
                total_n += 1
                if dataset.y[env_num][datapoint_num] == round(curr_prob):
                    acc += 1
        return acc / total_n
