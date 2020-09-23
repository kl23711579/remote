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
data_path = "/home/n10367071/remote/data/"

with open(data_path+"cluster_result.pkl", "rb") as f:
    cluster_result = pickle.load(f)

with open(data_path+"training_data.pkl", "rb") as f:
    X = pickle.load(f)

clusters = cluster_result[0]
inertia_vals = cluster_result[1]

plt.plot(range(2,30,1), inertia_vals, marker='*')
plt.savefig(data_path+"output.png")