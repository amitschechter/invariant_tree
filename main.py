# import os
#
# import argparse
# import numpy as np
#
# import utils
# from tree import Node
# from data import Dataset
#
#
# parser = argparse.ArgumentParser(description='inv tree')
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--th', type=float, default=0.5)
# parser.add_argument('--frac', type=float, default=0.9)
# parser.add_argument('--maxdepth', type=int, default=10)
# parser.add_argument('--nproj', type=int, default=10)
# parser.add_argument('--n_iter', type=int, default=10)
# flags = parser.parse_args()
#
# def print_args():
#     for k, v in sorted(vars(flags).items()):
#         print("\t{}: {}".format(k, v))
#
# def main():
#     print_args()
#
# if __name__ == '__main__':
#     main()
#
