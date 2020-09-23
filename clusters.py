import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import nltk
import string
import pickle

rs = 7

data_path = "/home/n10367071/remote/data/"

with open(data_path+"/Large_preprocess.pkl", "rb") as f:
    df = pickle.load(f)

df[["Latitude", 'Longitude', 'Altitude']] = pd.DataFrame(df["Point"].tolist(), index = df.index)

X = df[["Latitude", "Longitude", "Ryear"]].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# store training data -> X
with open(data_path+"training_data.pkl", "wb") as f:
    pickle.dump(X, f)

clusters = []
inertia_vals = []
for k in range(2,30,1):
    model = KMeans(n_clusters=k, random_state=rs)
    model.fit(X)
    
    clusters.append(model)
    inertia_vals.append(model.inertia_)

cluster_result = [clusters, inertia_vals]

with open(data_path+"cluster_result.pkl", "wb") as f:
    pickle.dump(cluster_result, f)







