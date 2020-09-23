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

def set_topic_prob(row, model, dictionary, best):
    c = row["Cluster_ID"]
    b = dictionary[c].doc2bow(preprocess(row["Rtitle"]))
    result = sorted(model[c][best[c]][b], key=lambda x: x[1], reverse=True)[0]
    return result

def preprocess(text):
    stemmer = SnowballStemmer('english')
    nltk_stopwords = stopwords.words('english')
    QUEENSLAND_RESOURCE_PERMIT = ['atp', 'daa', 'epc', 'epg', 'epm', 'epq', 'gl', 'mc', 'mdl', 
                                  'mfs', 'ml', 'oep', 'pca', 'pfl',' pga', 'pl', 'pp', 'ppl',
                                  'ppp', 'psa', 'psl', 'ql', 'wma']
    # gsq = GEOLOGICAL SURVEY OF QUEENSLAND
    OTHER_STOPWORDS = ['a-p', 'A-P', 'ap', 'AP', 'epp']
    nltk_stopwords.extend(QUEENSLAND_RESOURCE_PERMIT)
    nltk_stopwords.extend(OTHER_STOPWORDS)
    text = text.lower()
    words = [word for sent in sent_tokenize(text) for word in word_tokenize(sent) if word not in nltk_stopwords and len(word) > 1]
    tokens = []
    for word in words:
        # remove the word contains both digital and letters
        if re.search(r"\b[^\d\W]+\b", word):
            tokens.append(word)
        tokens = [stemmer.stem(word) for word in tokens ]
    return tokens

def preprocess2(text):
    nltk_stopwords2 = stopwords.words('english')
    QUEENSLAND_RESOURCE_PERMIT = ['atp', 'daa', 'epc', 'epg', 'epm', 'epq', 'gl', 'mc', 'mdl', 
                                  'mfs', 'ml', 'oep', 'pca', 'pfl',' pga', 'pl', 'pp', 'ppl',
                                  'ppp', 'psa', 'psl', 'ql', 'wma']
    # gsq = GEOLOGICAL SURVEY OF QUEENSLAND
    # after observing the result of LDA
    OTHER_STOPWORDS = ['a-p', 'A-P', 'ap', 'AP', 'epp', 'report', 'completion', "well", 
                       "period", "six", "area", "application", "annual", "monthly", "final",
                       "ended", "ending"]

    nltk_stopwords2.extend(QUEENSLAND_RESOURCE_PERMIT)
    nltk_stopwords2.extend(OTHER_STOPWORDS)
    text = text.lower()
    words = [word for sent in sent_tokenize(text) for word in word_tokenize(sent) if word not in nltk_stopwords2 and len(word) > 1]
    tokens = []
    for word in words:
        # remove the word contains both digital and letters
        if re.search(r"\b[^\d\W]+\b", word):
            tokens.append(word)
    return tokens

df = pd.read_csv("df_cluster.csv")

with open("LDA_models.pkl", "rb") as f:
    LDA_models = pickle.load(f)

with open("title_bow_s.pkl", "rb") as f:
    title_bow_s = pickle.load(f)

with open("title_dictionary_s.pkl", "rb") as f:
    title_dictionary_s = pickle.load(f)

with open("best.pkl", "rb") as f:
    best = pickle.load(f)

df["lda_result"] = df.apply(set_topic_prob, args=(LDA_models, title_dictionary_s, best), axis=1)
df[["Topic_ID", "Prob"]] = pd.DataFrame(df['lda_result'].tolist(), index=df.index)
df["Words"] = df["Rtitle"].apply(preprocess2)

df.to_csv("df_lda.csv")

with open("df_lda.pkl", "wb") as f:
    pickle.dump(df, f)