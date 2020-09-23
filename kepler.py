import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from keplergl import KeplerGl
import ast

with open("df_freq.pkl", "rb") as f:
    df = pickle.load(f)
    
df = df[["Latitude", "Longitude", "Ryear", "Cluster_ID", "Topic_ID", "freq_words"]]

df["Timestamp"] = df["Ryear"].apply(lambda x: datetime.timestamp(datetime(year=x, month=1, day=1)))

df = df.drop_duplicates()

cluster_topic_data = {}
clusters = np.unique(df["Cluster_ID"])
for cluster in clusters:
    df2 = df.loc[df["Cluster_ID"] == cluster]
    topicids = np.unique(df2["Topic_ID"])
    cluster_topic_data[cluster] = {}
    for topicid in topicids:
        s = f"topic_{topicid}"
        df3 = df2.loc[df2["Topic_ID"] == topicid]
        cluster_topic_data[cluster][s] = df3
     
for cluster in clusters:
    with open('./config/config_'+str(cluster)+'.txt', "r") as f:
        config = f.read()
    x = ast.literal_eval(config)
    w = KeplerGl(data=cluster_topic_data[cluster], config=x)
    file_name = f"./kepler/map_{cluster}.html"
    w.save_to_html(file_name=file_name)