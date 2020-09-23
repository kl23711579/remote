import pandas as pd
import numpy as np

import folium
from folium import Circle, CircleMarker

from collections import Counter

rs = 7

df = pd.read_csv("df_lda.csv")

clusters = np.unique(df["Cluster_ID"])

for cluster in clusters:
    g = df.loc[df["Cluster_ID"] == cluster].groupby(["Latitude", "Longitude"])
    g = list(g.groups.keys())
    lat0 = [ lat for lat, _ in g]
    lon0 = [ lon for _, lon in g]
    m = folium.Map(location=[-20.980330, 146.960505], zoom_start=5)
    for index in range(0, len(lat0)):
        folium.CircleMarker(location=[lat0[index], lon0[index]], radius=1).add_to(m)
    path = f"./vis/cluster_{cluster}.html"
    m.save(path)