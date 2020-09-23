import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from gensim.models import CoherenceModel

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import nltk
import string
import pickle

import matplotlib.pyplot as plt

import os, sys, re

rs = 7

with open("Large_preprocess2.pkl", "rb") as f:
    df = pickle.load(f)

X = df[["Latitude", "Longitude", "Ryear"]].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# store training data -> X
with open("training_data.pkl", "wb") as f:
    pickle.dump(X, f)

clusters = []
inertia_vals = []
for k in range(2,30,1):
    model = KMeans(n_clusters=k, random_state=rs)
    model.fit(X)
    
    clusters.append(model)
    inertia_vals.append(model.inertia_)

cluster_result = [clusters, inertia_vals]

with open("cluster_result.pkl", "wb") as f:
    pickle.dump(cluster_result, f)







