#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Experiments with UMAP  18-Oct-2022 J.Beale
# based on https://umap-learn.readthedocs.io/en/latest/basic_usage.html

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


# In[ ]:


cars = pd.read_csv("cars-7390.csv")  # read in CSV data
cars.head()


# In[ ]:


cars = cars.dropna()      # get rid of any N/A values
cars.hour.value_counts()  # display how many events at each hour of day


# In[ ]:


sns.pairplot(cars, vars=["v2", "std", "minY", "pixels"], hue='v2');  # slow


# In[ ]:


import umap
import umap.plot


# In[ ]:


cars_data = cars[
    [
        #"day",
        #"hour",
        #"minute",
        "frames",
        "v1",
        "v2",
        "v3",
        "std",
        "minY",
        "pixels",
    ]
].values
scaled_cars_data = StandardScaler().fit_transform(cars_data)


# In[ ]:


# embedding = reducer.fit_transform(scaled_cars_data) # this is a little slow
# embedding.shape


# In[ ]:


# plt.scatter(
#    embedding[:, 0],
#    embedding[:, 1]
#    )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of Oct.9-17 2022 Car Speed (v1-3)', fontsize=24);


# In[ ]:


mapper = umap.UMAP(n_neighbors=15,
                   min_dist=0.01,
                   init='spectral',
                   random_state=40).fit(scaled_cars_data) # this is a little slow


# In[ ]:


# this cell is a little slow

# umap.plot.diagnostic(mapper, diagnostic_type='vq')
# umap.plot.diagnostic(mapper, diagnostic_type='pca')
# umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
# umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')


# In[ ]:


def ishow(dmap, labels, n):
    hover_data = pd.DataFrame({'index': np.arange(n)+2,
                               'label': labels
                               #'x': dmap.embedding_[:n, 0],
                               #'y': dmap.embedding_[:n, 1]
                               })
    p = umap.plot.interactive(dmap, labels=labels,
                              hover_data=hover_data, point_size=3, theme = 'fire')
    umap.plot.show(p)
    
carA = cars.to_numpy()  # convert pandas dataframe to numpy array
img_count = carA.shape[0]   # how many total cars in data set
# labels = carA[:,6]          # v2 (average velocity)
#labels = carA[:,8]          # std (standard deviation of velocity)
#labels = carA[:,1]          # hour (hour of day, 24 hour clock)
#labels = carA[:,9]          # maxY (lowest vertical position in frame)
labels = carA[:,10]          # pixels (size of detected motion area)
ishow(mapper, labels, img_count)


# In[ ]:


print(carA.shape[0])
print(carA[0,5], carA[0,6], carA[0,7])

