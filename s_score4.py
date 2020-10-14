import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import string
import pickle

import os, sys

rs = 7
data_path = "/home/n10367071/remote/data/"

with open(data_path+"cluster_result.pkl", "rb") as f:
    cluster_result = pickle.load(f)

with open(data_path+"training_data.pkl", "rb") as f:
    X = pickle.load(f)

clusters = cluster_result[0]
inertia_vals = cluster_result[1]

# nums = [7,11,17,21]
nums = list(range(30,40,1))

for num in nums:
    print(f"Cluster {num}")
    score = silhouette_score(X, clusters[num-2].predict(X))
    print(f"Score for k={num} is {score}")