import random
import numpy as np
import torch

def gini(npos, n):
    p = npos / n if n > 0 else 0
    gini_score = n * p * (1 - p)
    return gini_score

def gini_normalized(npos, n, multiplier=1):
    p = npos / n if n > 0 else 0
    gini_score = multiplier * p * (1 - p)
    return gini_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

