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
plt.style.use('fivethirtyeight')

import os, sys, re
from collections import Counter

from df_freq import cluster_topic_freq_word_year_10

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
    
def preprocess_to_string(text):
    tokens = preprocess(text)
    return " ".join(tokens)

def lda_tfidf(num_topics, tfidf, text, dictionary, random_state, cluster_ID):
    coherence_ldas = []
    LDA_models = []
    topics = []
    for num_topic in num_topics:
        lda_tfidfmodel = models.LdaMulticore(tfidf, num_topics=num_topic, id2word=dictionary, passes=2, workers=2, random_state=random_state)
        coherence_model_lda = CoherenceModel(model=lda_tfidfmodel, texts=text, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        
        topics.append(num_topic)
        LDA_models.append(lda_tfidfmodel)
        coherence_ldas.append(coherence_lda) 
        
    plt.figure(figsize=(20,10))
    plt.plot(num_topics, coherence_ldas, marker='o', markersize=10)
    plt.savefig(f"{data_path}{cluster_ID}.png")
    
    best_index = coherence_ldas.index(max(coherence_ldas))
    # get best result
    # retuen model, topic_number
    return LDA_models, topics, best_index

def do_LDA(data, cluster_ID, min_topic_num=5, max_topic_num=31, steps=5, rs=7):
    title = data.loc[data['Cluster_ID'] == cluster_ID]
    process_title = title["Rtitle"].map(preprocess)
    title_dictionary = corpora.Dictionary(process_title)
    title_bow = [title_dictionary.doc2bow(title) for title in process_title]
    
    # calculate tfidf
    tfidf = models.TfidfModel(title_bow)
    title_tfidf = tfidf[title_bow]
    title_tfidf[0]
    
    LDA_models, topics, best_index = lda_tfidf(range(min_topic_num,max_topic_num,steps), title_tfidf, process_title, title_dictionary, rs, cluster_ID)
    print("Cluster {}, topic_number = {}".format(cluster_ID, topics[best_index]))

    return LDA_models, title_bow, title_dictionary, best_index

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

def set_topic_prob(row, model, dictionary, best):
    c = row["Cluster_ID"]
    b = dictionary[c].doc2bow(preprocess(row["Rtitle"]))
    result = sorted(model[c][best[c]][b], key=lambda x: x[1], reverse=True)[0]
    return result

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

df = pd.read_csv(data_path+"preprocess.csv")

clusters_number = 25
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

freq_word = cluster_topic_freq_word_year_10(df)

df["Freq_words"] = df.apply(get_freq_words, args=(freq_word, ),axis=1)

df.to_csv(data_path+"df_freq.csv", index=False)

with open(data_path+"LDA_models.pkl", "wb") as f:
	pickle.dump(LDA_models, f)

with open(data_path+"title_bow_s.pkl", "wb") as f:
	pickle.dump(title_bow_s, f)

with open(data_path+"title_dictionary_s.pkl", "wb") as f:
	pickle.dump(title_dictionary_s, f)

with open(data_path+"best.pkl", "wb") as f:
	pickle.dump(best, f)

with open(data_path+"df_lda.pkl", "wb") as f:
    pickle.dump(df, f)

with open(data_path+"df_freq.pkl", "wb") as f:
    pickle.dump(df, f)







