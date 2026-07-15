import os

import numpy as np
import pandas as pd

import datasets

n_features = 10


class RandTree:
    def __init__(self, x_c_temp=None):
        if x_c_temp is not None:
            self.x_c_temp = x_c_temp
            self.th = np.random.choice(self.x_c_temp)
        else:
            self.th = np.random.uniform(-10., 10)
        self.feature_ind = np.random.randint(n_features)
        self.is_leaf = False
        self.pred = np.random.uniform(0, 1)  # generates probability bet. 0 and 1
        self.left, self.right = None, None

    def build_tree(self, depth):
        # should be a leaf
        if depth <= 0:
            self.is_leaf = True
        else:
            x_c_temp = self.x_c_temp if self.x_c_temp is not None else None
            self.right = RandTree(x_c_temp)
            self.right.build_tree(depth - 1)
            self.left = RandTree(x_c_temp)
            self.left.build_tree(depth - 1)

    def predict(self, x):
        # for leaf node - return prediction
        if self.is_leaf:
            return self.pred

        # otherwise - recurse
        if x[self.feature_ind] <= self.th:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def print_info(self, depth, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces
        const = int(depth * width ** 1.5)
        spaces = "-" * const
        if not self.is_leaf:
            print(f"|{spaces} Split rule: x[{self.feature_ind}] <= {self.th}")
        if self.is_leaf:
            print(f"{' ' * const}   | Predicted prob: {self.pred}")

    def print_tree(self, depth):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info(depth)
        if self.left is not None:
            self.left.print_tree(depth + 1)
        if self.right is not None:
            self.right.print_tree(depth + 1)


def predict_from_x_with_tree(x_c, dummy_tree, n, dim, env):
    x_tmp = x_c + np.random.randn(n, dim) * env
    y = np.array([dummy_tree.predict(curr_x) for curr_x in x_tmp]).reshape(-1, 1)
    # y += np.random.randn(n, 1)
    return y

def generate_x_from_y(y, env, n, dim):
    x_nc = y.reshape(y.shape[0], 1) + np.random.randn(n, dim) #* env
    if np.random.choice(a=[0,1], p=[1-env, env]):
        x_nc = 1-x_nc
    return x_nc


def get_dataset_syn(n=100, dim=n_features, envs=[.9, .1, .95]):
    assert dim == n_features
    x_c_temp = np.concatenate([np.random.randn(n, dim) * env for env in envs])
    print(f'max: {np.max(x_c_temp)}, min: {np.min(x_c_temp)}')
    dummy_tree = RandTree(x_c_temp[:, 0])
    dummy_tree.build_tree(depth=2)
    dummy_tree.print_tree(depth=0)
    env_data = []
    for env in envs:
        x_c = np.random.randn(n, dim) * env
        y = predict_from_x_with_tree(x_c, dummy_tree, n, dim, env)
        x_nc = generate_x_from_y(y, env, n, dim)
        x = np.concatenate([x_c, x_nc], axis=1)
        y = np.array([int(curr_y[0] > 0.5) for curr_y in y])
        env_data.append({'x': x, 'y': y})

    env_dataset = datasets.DATASET_ENVS([env['x'] for env in env_data], [env['y'] for env in env_data])
    return env_dataset


def get_dataset_heart(dataset_folder='/home/amit/datasets/heart_failure_data',
                      all_files = ['cleveland.csv', 'hungarian.csv', 'switzerland.csv', 'va.csv']):

    col_names = ['age', 'sex', 'chest_pain', 'rest_bp', 'chol', 'fast_bp', 'rest_elect',
                 'max_hr', 'exercise_angina', 'oldpeak', 'slope', 'vessels_colored', 'thal', 'target_int']
    env_data = []
    for file_ind, filename in enumerate(all_files):
        filepath = os.path.join(dataset_folder, filename)
        curr_df = pd.read_csv(filepath, names=col_names)
        curr_y = (curr_df['target_int'] > 0).to_numpy().astype(int) # convert target to binary
        curr_df = curr_df.drop(labels=['target_int'], axis='columns')
        curr_X = curr_df.to_numpy()
        curr_X[curr_X == '?'] = -1
        curr_X = curr_X.astype(float).astype(int)
        env_data.append({'X': curr_X, 'y': curr_y})
    env_dataset = datasets.DATASET_ENVS([env['X'] for env in env_data], [env['y'] for env in env_data])
    return env_dataset


def preprocess_data(dataset):
    """
    :param dataset: list of environments, where each environment has X and y.
    :return: X, y: flatten environments, as well as n_examples, n_features.
    """
    X = np.concatenate([curr_env.X for j, curr_env in enumerate(dataset)])
    y = np.concatenate([curr_env.y for j, curr_env in enumerate(dataset)])
    n_examples, n_features = X.shape[0], X.shape[1]
    return X, y, n_examples, n_features


def split_env_dataset(dataset, val_env_ind):
    train_x = [x for index, x in enumerate(dataset.X) if index != val_env_ind]
    train_y = [y for index, y in enumerate(dataset.y) if index != val_env_ind]
    val_x, val_y = [dataset.X[val_env_ind]], [dataset.y[val_env_ind]]
    train_dataset = datasets.DATASET_ENVS(train_x, train_y)
    val_dataset = datasets.DATASET_ENVS(val_x, val_y)
    return train_dataset, val_dataset, train_x, train_y, val_x, val_y


def split_standard_dataset(dataset, val_env_ind):
    envs = dataset.datasets
    train_x = [env.X for index, env in enumerate(envs) if index != val_env_ind]
    train_y = [env.y for index, env in enumerate(envs) if index != val_env_ind]
    val_x, val_y = [envs[val_env_ind].X], [envs[val_env_ind].y]
    train_dataset = datasets.DATASET_ENVS(train_x, train_y)
    val_dataset = datasets.DATASET_ENVS(val_x, val_y)
    return train_dataset, val_dataset, train_x, train_y, val_x, val_y
