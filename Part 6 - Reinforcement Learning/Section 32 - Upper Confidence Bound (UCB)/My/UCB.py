# upper_confidence_bound

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import random
N = 10000
d = 10
ad_selected = []
number_of_rewards1 = [0] * d
number_of_rewards0 = [0] * d
for n in range(N):
    ad = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(number_of_rewards1[i] + 1, number_of_rewards0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i 
    ad_selected.append(ad)
    if dataset.values[n, ad] == 0:
        number_of_rewards0[ad] += 1
    else:
        number_of_rewards1[ad] += 1
    
total_reward = sum(number_of_rewards1)

# Plotting
plt.hist(ad_selected)
plt.show()