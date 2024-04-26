import torch
import numpy as np
import pickle
import pandas as pd

dir_alpha = .3

for x in np.random.dirichlet([dir_alpha] * 10):
    print(x)
