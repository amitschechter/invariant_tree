import argparse
import json
import os
import sys
import itertools

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import datasets
import inv_tree_model, inv_tree_test
from experiment_code import compare_invariance
from experiment_code.generate_data import get_dataset_syn, get_dataset_heart, \
    preprocess_data, split_env_dataset, split_standard_dataset
from lib import hparams_registry
from lib import misc
from utils import set_seed


def get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')
    # parser.add_argument('--data_dir', type=str, default='/home/amit/DomainBed/domainbed/data/MNIST/')
    parser.add_argument('--data_dir', type=str, default='/data/tml/code/as/tree/data/heart_failure_data/')

    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--n_proj', type=int, default=5,
                        help='Default number of projections to test')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--dataset', type=str, default="heart_failure",
                        choices=["RotatedMNIST", "ColoredMNIST", "synthetic", "heart_failure"])
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Max tree depth. Default is dataset-dependent.')
    parser.add_argument('--run_type', type=str, default="compare_trees",
                        choices=["analyze_tree", "compare_trees", "hyperparameters"])
    args = parser.parse_args()
    return args


def get_hparams(args):
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))
    return hparams

def analyze_tree(dataset, args):
    """Train standard tree and check how invariant it is"""
    X, y, n_examples, n_features = preprocess_data(dataset)
    clf = DecisionTreeClassifier(max_depth=args.max_depth).fit(X, y)
    print(f'train acc: {clf.score(X, y)}')
    compare_invariance.compare_inv(clf, dataset)
    print(X, np.min(X), np.max(X))

def run_inv_tree_comp(dataset, args, val_env_ind_arr = None):
    """Train standard decision tree/RF and compare to inv tree"""
    # if val_env_ind_arr is empty - iterate over all possible env as val
    if val_env_ind_arr is None:
        val_env_ind_arr = np.arange(dataset.n_envs)

    for val_env_ind in val_env_ind_arr:
        print(f'\nCurr val env index: {val_env_ind}')
        if args.dataset == 'synthetic' or args.dataset == 'heart_failure':
            train_dataset, val_dataset, train_x, train_y, val_x, val_y = split_env_dataset(
                dataset, val_env_ind=val_env_ind)
        else:
            train_dataset, val_dataset, train_x, train_y, val_x, val_y = split_standard_dataset(
                dataset, val_env_ind=val_env_ind)

        # standard sklearn tree
        clf = DecisionTreeClassifier(max_depth=args.max_depth).fit(
            np.concatenate(train_x), np.concatenate(train_y))
        print('sklearn tree')
        print(f'  train: {clf.score(np.concatenate(train_x), np.concatenate(train_y))}')
        print(f'  val: {clf.score(val_x[0], val_y[0])}')
        sklearn.tree.plot_tree(clf)
        plt.savefig('/afs/csail.mit.edu/u/a/as/fig1.pdf')

        # standard random forest classifieir
        clf = RandomForestClassifier(n_estimators=10, max_depth=args.max_depth).fit(
            np.concatenate(train_x), np.concatenate(train_y))
        print('sklearn random forest')
        print(f'  train: {clf.score(np.concatenate(train_x), np.concatenate(train_y))}')
        print(f'  val: {clf.score(val_x[0], val_y[0])}')


        # invariant tree model with lambda = 0 (i.e. not invariant):
        #tree = inv_tree_model.Node(lambda_val=0.0)
        tree = inv_tree_test.Node(lambda_val=0.0)
        tree.build(train_dataset, max_depth=args.max_depth, n_proj=0, index_arr=None)
        print('inv tree with lambda=0 (i.e. not inv)')
        print(f'  train acc: {tree.eval(train_dataset)}')
        print(f'  val acc: {tree.eval(val_dataset)}')

        # invariant tree model
        lambda_val = 1000.0
        # tree = inv_tree_model.Node(lambda_val=lambda_val)
        tree = inv_tree_test.Node(lambda_val=lambda_val)
        tree.build(train_dataset, max_depth=args.max_depth, n_proj=0, index_arr=None)
        # print('inv tree with lambda=1000')
        print(f'inv tree with lambda={lambda_val}')
        print(f'  train acc: {tree.eval(train_dataset)}')
        print(f'  val acc: {tree.eval(val_dataset)}')
        # tree.print_tree(depth=1)

def run_hyperparams(dataset, args, val_env_ind_arr = None):
    if val_env_ind_arr is None:
        val_env_ind_arr = np.arange(dataset.n_envs)
    
    lambda_val = [0.1, 1, 10, 100, 1000, 10000]
    tree_depth = [1,2,3,4,5,6,7,8]
    avg_val_lst = []

    
    for combo in itertools.product(lambda_val, tree_depth):
        curr_lambda, curr_depth = combo[0], combo[1]
        train_acc_lst, val_acc_lst = [], []
        for val_env_ind in val_env_ind_arr:
            print(f'\nCurr val env index: {val_env_ind}')
            if args.dataset == 'synthetic' or args.dataset == 'heart_failure':
                    train_dataset, val_dataset, train_x, train_y, val_x, val_y = split_env_dataset(
                        dataset, val_env_ind=val_env_ind)
            else:
                train_dataset, val_dataset, train_x, train_y, val_x, val_y = split_standard_dataset(
                    dataset, val_env_ind=val_env_ind)
                    
            tree = inv_tree_test.Node(lambda_val=curr_lambda)
            tree.build(train_dataset, max_depth=curr_depth, n_proj=0, index_arr=None)

            print(f'inv tree with lambda={curr_lambda}, depth: {curr_depth}, val env index: {val_env_ind}')
            train_acc, val_acc = tree.eval(train_dataset), tree.eval(val_dataset)
            train_acc_lst.append(train_acc)
            val_acc_lst.append(val_acc)
            print(f'  train acc: {train_acc}')
            print(f'  val acc: {val_acc}')
        avg_val_lst.append(np.mean(val_acc_lst))
        print(f'lambda={curr_lambda}, depth: {curr_depth}, avg_val_acc: {np.mean(val_acc_lst)}, avg_train_acc: {np.mean(train_acc_lst)} \n\n')
    best_acc, best_ind = np.max(avg_val_lst), np.argmax(avg_val_lst)
    print(f'best acc: {best_acc}')
    combo = list(itertools.product(lambda_val, tree_depth))[best_ind]
    print(f'best_ind: {best_ind}, best_params: {combo}')

def run_hyperparams_forest(dataset, args, val_env_ind_arr = None):
    if val_env_ind_arr is None:
        val_env_ind_arr = np.arange(dataset.n_envs)
    
    n_estimators = [5,10,50,100]
    tree_depth = [1,2,3,4,5,6,7,8]
    avg_val_lst = []

    
    for combo in itertools.product(n_estimators, tree_depth):
        curr_n_estimators, curr_depth = combo[0], combo[1]
        train_acc_lst, val_acc_lst = [], []
        for val_env_ind in val_env_ind_arr:
            print(f'\nCurr val env index: {val_env_ind}')
            if args.dataset == 'synthetic' or args.dataset == 'heart_failure':
                    train_dataset, val_dataset, train_x, train_y, val_x, val_y = split_env_dataset(
                        dataset, val_env_ind=val_env_ind)
            else:
                train_dataset, val_dataset, train_x, train_y, val_x, val_y = split_standard_dataset(
                    dataset, val_env_ind=val_env_ind)
                    
            clf = RandomForestClassifier(n_estimators=curr_n_estimators, max_depth=curr_depth).fit(
                np.concatenate(train_x), np.concatenate(train_y))

            print(f'inv tree with lambda={curr_n_estimators}, depth: {curr_depth}, val env index: {val_env_ind}')
            train_acc, val_acc = clf.score(np.concatenate(train_x), np.concatenate(train_y), clf.score(val_x[0], val_y[0])) #tree.eval(train_dataset), tree.eval(val_dataset)
            train_acc_lst.append(train_acc)
            val_acc_lst.append(val_acc)
            print(f'  train acc: {train_acc}')
            print(f'  val acc: {val_acc}')
        avg_val_lst.append(np.mean(val_acc_lst))
        print(f'lambda={curr_n_estimators}, depth: {curr_depth}, avg_val_acc: {np.mean(val_acc_lst)}, avg_train_acc: {np.mean(train_acc_lst)} \n\n')
    best_acc, best_ind = np.max(avg_val_lst), np.argmax(avg_val_lst)
    print(f'best acc: {best_acc}')
    combo = list(itertools.product(n_estimators, tree_depth))[best_ind]
    print(f'best_ind: {best_ind}, best_params: {combo}')


if __name__ == "__main__":
    args = get_args()
    print('\nArgs:\n', '\n'.join(f'{k}={v}' for k, v in vars(args).items()), '\n')

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))
    hparams = get_hparams(args)
    set_seed(args.seed)

    if args.dataset in vars(datasets):
        """
        dataset has:
            X: list of x entries for the different environments
            y: list of y entries for the different environments
            access as follows: dataset.X, dataset.y
        """
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    elif args.dataset == 'synthetic':
        """
        dataset has:
            x: list of x entries for the different environments
            y: list of y entries for the different environments
        """
        dataset = get_dataset_syn()
    elif args.dataset == 'heart_failure':
        """ Same dataset structure as synthetic one """
        dataset = get_dataset_heart(args.data_dir)
    else:
        raise NotImplementedError

    # Train sklearn tree and check how invariant it is
    if args.run_type == 'analyze_tree':
        analyze_tree(dataset, args)
    elif args.run_type == 'compare_trees':
        run_inv_tree_comp(dataset, args)
    elif args.run_type == 'hyperparameters':
        run_hyperparams(dataset, args)
    else:
        raise NotImplementedError
