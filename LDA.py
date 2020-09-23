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

import nltk
import string
import pickle

import matplotlib.pyplot as plt

import os, sys, re

rs = 7
data_path = "/home/n10367071/remote/data/"

stemmer = SnowballStemmer('english')
nltk_stopwords = stopwords.words('english')
QUEENSLAND_RESOURCE_PERMIT = ['atp', 'daa', 'epc', 'epg', 'epm', 'epq', 'gl', 'mc', 'mdl', 
                            'mfs', 'ml', 'oep', 'pca', 'pfl',' pga', 'pl', 'pp', 'ppl',
                            'ppp', 'psa', 'psl', 'ql', 'wma']
# gsq = GEOLOGICAL SURVEY OF QUEENSLAND
OTHER_STOPWORDS = ['a-p', 'A-P', 'ap', 'AP', 'epp']
nltk_stopwords.extend(QUEENSLAND_RESOURCE_PERMIT)
nltk_stopwords.extend(OTHER_STOPWORDS)

def preprocess(text):
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
    plt.close()
    
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

 

# reverse_result
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

def preprocess2(text):
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