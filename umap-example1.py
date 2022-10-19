#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Experiments with UMAP  19-Oct-2022 J.Beale
# based on https://umap-learn.readthedocs.io/en/latest/basic_usage.html

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


# In[ ]:


cars  = pd.read_csv("cars-7390.csv")  # Get training data from CSV file
# cars.head()


# In[ ]:


# cars = cars.dropna()      # get rid of any N/A values
# cars.hour.value_counts()  # display how many events at each hour of day


# In[ ]:


# sns.pairplot(cars, vars=["v2", "std", "minY", "pixels"], hue='v2');  # slow


# In[ ]:


import umap                 # this cell takes a while
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

# fit_transform(X) calculates the best fit, then transforms the data
# fit(X) just calculates the parameters (per-column) returning the scaler object for later use

Fit = StandardScaler().fit(cars_data)  # get parameters needed to standardize this data
scaled_cars_data = Fit.transform(cars_data)


# In[ ]:


mapper = umap.UMAP(n_neighbors=15,
                   min_dist=0.01,
                   init='spectral',
                   random_state=40).fit(scaled_cars_data) # this is a little slow


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

#labels = carA[:,1]          # hour (hour of day, 24 hour clock)
#labels = carA[:,6]          # v2 (average velocity)
#labels = carA[:,8]          # std (standard deviation of velocity)
#labels = carA[:,9]          # maxY (lowest vertical position in frame)
labels = carA[:,10]          # pixels (size of detected motion area)

ishow(mapper, labels, img_count)  # show an interactive plot of the training data


# In[ ]:


orig_embedding = mapper.transform(scaled_cars_data)   # original training data in map
plt.scatter(orig_embedding[:, 0], orig_embedding[:, 1], c=carA[:,6], s=2, cmap='Spectral') # plot training data


# In[ ]:


# === Now, let's load new test data, and see how it fits into the map


# In[ ]:


cars2 = pd.read_csv("cars-1525.csv")  # Get test data from CSV file


# In[ ]:


cars2_data = cars2[
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

scaled_cars2_data = Fit.transform(cars2_data)  # transform test data with existing trained Fit parameters


# In[ ]:


test_embedding = mapper.transform(scaled_cars2_data)  # put test data into trained map


# In[ ]:


# Display the new data based on the trained UMAP embedding

# index: 1 2 3 4 5  6  7  8   9    10
# value: D H M S v1 v2 v3 std minY size

car2A = cars2.to_numpy()  # convert pandas dataframe to numpy array
plt.rcParams['axes.facecolor'] = 'black'  # set matplotlib background color

#plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=car2A[:,10], cmap='Spectral')
#plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=car2A[:,9], cmap='Spectral')
#plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=car2A[:,8], cmap='Spectral')
plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=car2A[:,6], s=2, cmap='Spectral')


# In[ ]:




