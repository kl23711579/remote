import pickle
import pandas as pd
import numpy as np

from collections import Counter

rs = 7

with open("df_lda.pkl", "rb") as f:
    df = pickle.load(f)

clusters = np.unique(df["Cluster_ID"])

cluster_freq_words = []
for cluster in clusters:
    df2 = df.loc[df["Cluster_ID"] == cluster]
    words = df2["Words"].tolist()
    word = [ i for x in words for i in x]
    common = [i for i in Counter(word).most_common(10)]
    cluster_freq_words.append(common)

with open("./freq_words/cluster_freq_words.pkl", "wb") as f:
	pickle.dump(cluster_freq_words, f)

cluster_freq_words_years = []
for cluster in clusters:
    df2 = df.loc[df["Cluster_ID"] == cluster]
    years = np.unique(df2["Ryear"])
    cluster_freq_words_years.append({})
    for year in years:
        df3 = df2.loc[df2["Ryear"] == year]
        words = df3["Words"].tolist()
        word = [ i for x in words for i in x]
        common = [i for i in Counter(word).most_common(10)]
        cluster_freq_words_years[cluster][year] = common

with open("./freq_words/cluster_freq_words_years.pkl", "wb") as f:
	pickle.dump(cluster_freq_words_years, f)

cluster_topic_freq_words = []
for cluster in clusters:
    df2 = df.loc[df["Cluster_ID"] == cluster]
    topicids = np.unique(df2["Topic_ID"])
    cluster_topic_freq_words.append({})
    for topicid in topicids:
        df3 = df2.loc[df2["Topic_ID"] == topicid]
        words = df3["Words"].tolist()
        word = [ i for x in words for i in x]
        common = [i for i in Counter(word).most_common(10)]
        cluster_topic_freq_words[cluster][topicid] = common

with open("./freq_words/cluster_topic_freq_words.pkl", "wb") as f:
	pickle.dump(cluster_topic_freq_words, f)


cluster_topic_freq_words_years = []
clusters = np.unique(df["Cluster_ID"])
for cluster in clusters:
    df2 = df.loc[df["Cluster_ID"] == cluster]
    topicids = np.unique(df2["Topic_ID"])
    cluster_topic_freq_words_years.append({})
    for topicid in topicids:
        df3 = df2.loc[df2["Topic_ID"] == topicid]
        years = np.unique(df3["Ryear"])
        cluster_topic_freq_words_years[cluster][topicid] = {}
        for year in years:
            words = df3.loc[df3["Ryear"] == year]["Words"].tolist()
            a1 = [ i for x in words for i in x]
            word = [ i for x in words for i in x]
            common = [i for i in Counter(word).most_common(10)]
            cluster_topic_freq_words_years[cluster][topicid][year] = common

with open("./freq_words/cluster_topic_freq_words_years.pkl", "wb") as f:
	pickle.dump(cluster_topic_freq_words_years, f)