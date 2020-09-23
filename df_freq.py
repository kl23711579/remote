import pandas as pd
import numpy as np
import pickle

with open("df_lda.pkl", "rb") as f:
    df = pickle.load(f)

with open("./freq_words/cluster_topic_freq_words_years.pkl", "rb") as f:
    cluster_topic_freq_words_years = pickle.load(f)
            
def get_freq_words(row, words):
    cluster = row["Cluster_ID"]
    topic = row["Topic_ID"]
    year = row["Ryear"]
    w = ""
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
    return w

df["freq_words"] = df.apply(get_freq_words, args=(cluster_topic_freq_words_years, ),axis=1)

df.to_csv("df_freq.csv")
with open("df_freq.pkl", "wb") as f:
    pickle.dump(df, f)