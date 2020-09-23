import pandas as pd
import numpy as np

import string
import pickle

import os, sys, re

from Freq_words import cluster_topic_freq_word_year_10
from LDA import do_LDA, preprocess2, set_topic_prob

rs = 7
data_path = "/home/n10367071/remote/data/"

def get_clusters(df, cluster_number):
    with open(data_path+"cluster_result.pkl", "rb") as f:
        cluster_result = pickle.load(f)

    clusters = cluster_result[0]

    with open(data_path+"training_data.pkl", "rb") as f:
        X = pickle.load(f)

    y = clusters[cluster_number-2].predict(X)

    df["Cluster_ID"] = y

    with open(data_path+"df_cluster.pkl", "wb") as f:
        pickle.dump(df, f)

    return df

def get_freq_words(row, words):
    cluster = row["Cluster_ID"]
    topic = row["Topic_ID"]
    year = row["Ryear"]
    w = ""
    try:
        if len(words[cluster][topic][year]) >= 3:
            word1 = words[cluster][topic][year][0][0]
            word2 = words[cluster][topic][year][1][0]
            word3 = words[cluster][topic][year][2][0]
            w = word1 + ", " + word2 + ", " + word3
        elif len(words[cluster][topic][year]) == 2:
            word1 = words[cluster][topic][year][0][0]
            word2 = words[cluster][topic][year][1][0]
            w = word1 + ", " + word2
        elif len(words[cluster][topic][year]) == 1:
            w = words[cluster][topic][year][0][0]
    except:
        w = ""
    return w

with open(data_path+"Large_preprocess.pkl", "rb") as f:
    df = pickle.load(f)

clusters_number = 14
df = get_clusters(df, clusters_number)

LDA_models = []
title_bow_s = []
title_dictionary_s = []
best=[]

for cluster_number in range(0, clusters_number):
    LDA_model, title_bow, title_dictionary, best_index = do_LDA(df, cluster_number, 4, 37, 2)
    LDA_models.append(LDA_model)
    title_bow_s.append(title_bow)
    title_dictionary_s.append(title_dictionary)
    best.append(best_index)

df["LDA_result"] = df.apply(set_topic_prob, args=(LDA_models, title_dictionary_s, best), axis=1)
df[["Topic_ID", "Prob"]] = pd.DataFrame(df['LDA_result'].tolist(), index=df.index)
df["Words"] = df["Rtitle"].apply(preprocess2)

df.to_csv(data_path+"df_lda.csv", index=False)
with open(data_path+"df_lda.pkl", "wb") as f:
    pickle.dump(df, f)

freq_word = cluster_topic_freq_word_year_10(df)

df["Freq_words"] = df.apply(get_freq_words, args=(freq_word, ),axis=1)

df.to_csv(data_path+"df_final.csv", index=False)

print("Work Successful!")

with open(data_path+"LDA_models.pkl", "wb") as f:
	pickle.dump(LDA_models, f)

with open(data_path+"title_bow_s.pkl", "wb") as f:
	pickle.dump(title_bow_s, f)

with open(data_path+"title_dictionary_s.pkl", "wb") as f:
	pickle.dump(title_dictionary_s, f)

with open(data_path+"best.pkl", "wb") as f:
	pickle.dump(best, f)

with open(data_path+"df_final.pkl", "wb") as f:
    pickle.dump(df, f)







