import pickle
import pandas as pd
import numpy as np

from collections import Counter

rs = 7
data_path = "/home/n10367071/remote/data/"

def cluster_freq_word(df=None):
    '''
    Count freq words depends on Cluster.
    '''
    clusters = np.unique(df["Cluster_ID"])

    cluster_freq_word = []
    for cluster in clusters:
        df2 = df.loc[df["Cluster_ID"] == cluster]
        words = df2["Words"].tolist()
        word = [ i for x in words for i in x]
        # common = [i for i in Counter(word).most_common(10)]
        common = dict(Counter(word))
        cluster_freq_word.append(common)

    with open(data_path+"freq_words/cluster_freq_word.pkl", "wb") as f:
        pickle.dump(cluster_freq_word, f)

def cluster_freq_word_year(df=None):
    '''
    Count freq words depends on cluster and year.
    '''
    clusters = np.unique(df["Cluster_ID"])
    cluster_freq_words_year = []
    for cluster in clusters:
        df2 = df.loc[df["Cluster_ID"] == cluster]
        years = np.unique(df2["Ryear"])
        cluster_freq_words_year.append({})
        for year in years:
            df3 = df2.loc[df2["Ryear"] == year]
            words = df3["Words"].tolist()
            word = [ i for x in words for i in x]
            common = dict(Counter(word))
            cluster_freq_words_year[cluster][year] = common

    with open(data_path+"freq_words/cluster_freq_word_year.pkl", "wb") as f:
        pickle.dump(cluster_freq_words_year, f)

def cluster_topic_freq_word(df=None):
    '''
    Count freq word base on cluster and topic.
    '''
    clusters = np.unique(df["Cluster_ID"])
    cluster_topic_freq_word = []
    for cluster in clusters:
        df2 = df.loc[df["Cluster_ID"] == cluster]
        topicids = np.unique(df2["Topic_ID"])
        cluster_topic_freq_word.append({})
        for topicid in topicids:
            df3 = df2.loc[df2["Topic_ID"] == topicid]
            words = df3["Words"].tolist()
            word = [ i for x in words for i in x]
            common = dict(Counter(word))
            cluster_topic_freq_word[cluster][topicid] = common

    with open(data_path+"freq_words/cluster_topic_freq_word.pkl", "wb") as f:
        pickle.dump(cluster_topic_freq_word, f)


def cluster_topic_freq_word_year(df=None): 
    '''
    Count freq word base on cluster topic and year.
    '''
    cluster_topic_freq_word_year = []
    clusters = np.unique(df["Cluster_ID"])
    for cluster in clusters:
        df2 = df.loc[df["Cluster_ID"] == cluster]
        topicids = np.unique(df2["Topic_ID"])
        cluster_topic_freq_word_year.append({})
        for topicid in topicids:
            df3 = df2.loc[df2["Topic_ID"] == topicid]
            years = np.unique(df3["Ryear"])
            cluster_topic_freq_word_year[cluster][topicid] = {}
            for year in years:
                words = df3.loc[df3["Ryear"] == year]["Words"].tolist()
                word = [ i for x in words for i in x]
                common = dict(Counter(word))
                cluster_topic_freq_word_year[cluster][topicid][year] = common

    with open(data_path+"freq_words/cluster_topic_freq_word_year.pkl", "wb") as f:
        pickle.dump(cluster_topic_freq_word_year, f)

def cluster_topic_freq_word_year_10(df=None): 
    '''
    Count freq word base on cluster topic and year.
    '''
    cluster_topic_freq_word_year = []
    clusters = np.unique(df["Cluster_ID"])
    for cluster in clusters:
        df2 = df.loc[df["Cluster_ID"] == cluster]
        topicids = np.unique(df2["Topic_ID"])
        cluster_topic_freq_word_year.append({})
        for topicid in topicids:
            df3 = df2.loc[df2["Topic_ID"] == topicid]
            years = np.unique(df3["Ryear"])
            cluster_topic_freq_word_year[cluster][topicid] = {}
            for year in years:
                words = df3.loc[df3["Ryear"] == year]["Words"].tolist()
                word = [ i for x in words for i in x]
                common = [i for i in Counter(word).most_common(10)]
                cluster_topic_freq_word_year[cluster][topicid][year] = common

    return cluster_topic_freq_word_year

if __name__ == "__main__":
    with open(data_path+"df_lda.pkl", "rb") as f:
        df = pickle.load(f)

    cluster_freq_word(df)
    cluster_freq_word_year(df)
    cluster_topic_freq_word(df)
    cluster_topic_freq_word_year(df)
