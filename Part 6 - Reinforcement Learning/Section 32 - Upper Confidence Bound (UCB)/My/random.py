# random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
total = 0
n = 10000
d = 10
for i in range(n):
    ad = random.randrange(d)
    total += dataset.values[i, ad]
    