import numpy as np
import utils

class Node:
    def __init__(self):
        self.dim = 0
        self.leaf = True
        self.th, self.prob = 0, 0.5
        self.projW = None
        self.left, self.right = None, None

    def predict(self, x_arr):
        if self.leaf:
            return self.prob
        x_val = np.sum(np.array(x_arr) * self.projW)
        if x_val < self.th:
            return self.left.predict(x_arr)
        else:
            return self.right.predict(x_arr)

    def eval(self, env):
        acc = 0
        for i, curr_x in enumerate(env.X):
            p = self.predict(curr_x)
            if (env.y[i] - 0.5) * (p - 0.5) > 0:
                acc += 1
        return acc / env.n_examples

    def add_eval(self, env, pred):
        for i in range(env.n_examples):
            pred[i] += self.predict(env.X[i])

    def gini_gain(self, x, y, w):
        n = len(x)
        assert n > 1, "n must be greater than 1"
        sorted_index_arr = np.argsort(x)

        # w_total, w_pos = 0, 0
        # for curr_index in sorted_index_arr:
        #     w_total += w[curr_index]
        #     if y[curr_index] == 1:
        #         w_pos += w[curr_index]
        w_total = np.sum(w[sorted_index_arr])
        w_pos = np.sum(y[sorted_index_arr] * w[sorted_index_arr])
        H0 = utils.gini(w_pos, w_total)

        gain, w_pos_cum, w_total_cum, th = 0, 0, 0, 0
        for i in range(n - 1):
            curr_index = sorted_index_arr[i]
            w_total_cum += w[curr_index]
            w_pos_cum += w[curr_index] * y[curr_index]
            # if possible to split in between
            if x[sorted_index_arr[i+1]] > x[sorted_index_arr[i]]:
                gain_tmp = H0
                gain_tmp -= utils.gini(w_pos_cum, w_total_cum)
                gain_tmp -= utils.gini(w_pos - w_pos_cum, w_total - w_total_cum)

                if gain_tmp > gain:
                    gain = gain_tmp
                    th = (x[sorted_index_arr[i+1]] + x[sorted_index_arr[i]]) / 2
        return gain, th

    def build(self, env, max_depth, n_proj, index):
        dim = env.n_features
        if index is None:
            index = np.arange(len(env.y))
        n = len(index)

        # prediction at the node is a weighted average
        wtot = np.sum(env.w[index])
        self.prob = np.sum(env.w[index] * env.y[index])
        self.prob /= wtot

        if max_depth <= 0 or n <= 1:
            return

        self.leaf, self.th, gain, ind = True, 0, 0, 0
        Z = np.random.normal(size=(n_proj, dim))
        # Z = np.identity(dim)

        for k in range(n_proj):
        # for k, curr_proj in enumerate(Z):
            x = np.sum(Z[k].reshape(1, -1) * env.X[index], axis=1)
            y = env.y[index]
            w = env.w[index]
            gain_tmp, th_tmp = self.gini_gain(x, y, w)
            if gain_tmp > gain:
                self.leaf = False
                gain = gain_tmp
                ind = k
                self.th = th_tmp

        # store the selected projection
        if gain > 0:
            self.projW = Z[ind]
            x = np.sum(self.projW * env.X[index], axis=1)
            mask_left = x < self.th
            sel_left, sel_right = index[mask_left], index[~mask_left]
            assert len(sel_left) > 0 and len(sel_right) > 0  # if not, should be a leaf
            self.left = Node()
            self.left.build(env, max_depth-1, n_proj, sel_left)
            self.right = Node()
            self.right.build(env, max_depth-1, n_proj, sel_right)

    def print_info(self, depth, width=4):
        """
        Method to print the information about the tree
        """
        # Defining the number of spaces for indentation
        const = int(depth * width ** 1.5)
        spaces = "-" * const

        print(f"|{spaces} Split rule: x{np.where(self.projW != 0)} < {self.th}")
        if self.leaf:
            print(f"{' ' * const}  | Predicted prob: {self.prob}")

    def print_tree(self, depth):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info(depth)
        if self.left is not None:
            self.left.print_tree(depth + 1)
        if self.right is not None:
            self.right.print_tree(depth + 1)

class Ensemble:
    def __init__(self, n):
        self.n = n
        self.T = np.empty(n)
        for i in range(n):
            self.T[i] = Node()

    def build(self, env, max_depth, n_proj):
        assert env.n_examples > 0
        index = np.arange(env.n_examples)
        for i in range(self.n):
            self.T[i].build(env, max_depth, n_proj, index, env.n_examples) # TODO: check if index is modified
            print(f'{i} ', end='')
        print('')

    def eval(self, env):
        assert env.n_examples > 0
        psum = np.zeros(env.n_examples)
        for i in range(self.n):
            self.T[i].add_eval(env, psum)

        acc = 0
        for i in range(env.n_examples):
            acc += (env.y[i]-0.5) * psum[i]-self.n * 0.5
        return acc / env.n_examples

