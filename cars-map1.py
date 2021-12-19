# -*- coding: utf-8 -*-
"""
UMAP plot for input images clustered according to similarity.

Created on Sun Dec 12 11:28:45 2021
@author: jbeale
https://umap-learn.readthedocs.io/en/latest/parameters.html

"""

import numpy as np
import seaborn as sns
import umap
import umap.plot
from math import sqrt
import os  # loop over images in directory
import skimage.io  # to read & save images
from skimage.util import img_as_float
import pandas as pd
import showPlot  # local lib to display a set of images, given filenames

# import datashader as ds
# import plotly.express as px
# from plotly.offline import plot

# ---------------------------------------------------------------
#  cartesian distance between two points (a,b are 2-elem vectors)


def jdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dist = sqrt(dx*dx + dy*dy)
    return(dist)


# -------------------------------------------------------
#  find index and distance of the nearest elment in ar[] to point p


def jmin(p, ar):
    mdist = jdist(p, ar[0])
    nearest = 0
    for i in range(len(ar)):
        dist = jdist(p, ar[i])
        if (dist < mdist):
            mdist = dist
            nearest = i
    return(nearest, mdist)


# -----------------------------------------------

path = "C:\\Users\\beale\\Documents\\Umap\\raw2"  # raw input images here
#img_count = 10000   # how many images to consider
#img_count = 21669   # how many images to consider
px_count = 53*20    # how many pixels in each image (x * y)
dt = np.dtype(('U', 25))  # string with filename minus extension

sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})
# data = np.zeros((img_count, px_count), dtype=np.float32)  # items,parts of item
# pname = np.ndarray(img_count, dt)


"""
# Loop over (img_count) input image files and add them into data[]
# also add the filenames to pname[]
i = 0  # item index
for iname in os.listdir(path):
    if (iname.startswith("DH5_")) and (iname.lower().endswith(".png")):
        fname_in = os.path.join(path, iname)
        if (i%1000) == 0:
            print(i, fname_in)
        pname[i] = (iname[0:25])
        img = img_as_float(skimage.io.imread(fname=fname_in))  # img input
        data[i, :] = img.reshape(px_count)
        i = i + 1
        if (i >= img_count):
            break
"""
data = np.load("data_6413.npy")
pname = np.load("pname_6413.npy")
img_count = len(pname)

# np.random.seed(17)  # dens_lambda=1, min_dist=0.1
np.random.seed(42)
# inpos = mapper.embedding_
# --- Generate the clustered similarity map for the data
mapper = umap.UMAP(
    # densmap=True, dens_lambda=2, spread=1,
    n_neighbors=10, min_dist=0.014,
    #  init=inpos,
    init='spectral',
    random_state=42,
    ).fit(data)
umap.plot.diagnostic(mapper, diagnostic_type='vq')

# umap.plot.points(mapper)
# umap.plot.connectivity(mapper, show_points=True)  # inter-connection lines
# umap.plot.diagnostic(mapper, diagnostic_type='pca')
# local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
# umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')

# display plot on an interactive web page


def ishow(dmap, picname, n):
    hover_data = pd.DataFrame({'index': np.arange(n),
                               'label': picname[:n],
                               'x': dmap.embedding_[:n, 0],
                               'y': dmap.embedding_[:n, 1]
                               })
    p = umap.plot.interactive(dmap, labels=picname[:n],
                              hover_data=hover_data, point_size=3)
    umap.plot.show(p)


ishow(mapper, pname, img_count)
# ishow(mapper1,pname1,n_ROI)


"""
# for seed(42), dlambda=1.0, neigbors=10, mindist=0.1, spread=1.0

# for seed(42), dlambda=2.0, neigbors=10, mindist=0.01, spread=1.0
inpos = mapper.embedding_
xr = [8.45, 21]
yr = [2.7, 16.5 ]
j = 0
data1 = np.zeros((n_ROI, px_count), dtype=np.float32)  # ROI only
pname1 = np.ndarray(n_ROI, dt)  # filenames in ROI
for i in range(len(inpos)):
    p = inpos[i]
    if p[0]>xr[0] and p[0]<xr[1] and p[1]>yr[0] and p[1]<yr[1]:
        print("%d,%d,%5.3f,%5.3f,%s" %(j, i, p[0], p[1], pname[i]))
        data1[j, :] = data[i, :]
        pname1[j] = pname[i]  # save the ROI name from original names
        j += 1
n_ROI = j  # count of data items within ROI



i_cent = 8888   # index of item in center of ROI
i_edge = 8612   # index of item at edge of ROI
d_center = mapper.embedding_[i_cent]  # coords of data at center ROI
r_active = jdist(d_center, mapper.embedding_[i_edge])  # radius of ROI

j = 0
for i in range(len(mapper.embedding_)):
    r_test = jdist(d_center, mapper.embedding_[i])
    if (r_test < r_active):
        print(j, i, r_test, pname[i])
        j += 1

n_ROI = j  # count of data items within ROI
data1 = np.zeros((n_ROI, px_count), dtype=np.float32)  # ROI only
# imap1 = np.zeros(n_ROI, dtype=np.int32)  # map new index to original index
pname1 = np.ndarray(n_ROI, dt)  # filenames in ROI
j = 0
for i in range(len(mapper.embedding_)):
    r_test = jdist(d_center, mapper.embedding_[i])
    if (r_test < r_active):
        data1[j, :] = data[i, :]
        pname1[j] = pname[i]  # save the ROI name from original names
        j += 1

# mapper1 = umap.UMAP(n_neighbors=10, min_dist=0.05).fit(data1)
# mapper1 = umap.UMAP(n_neighbors=10, min_dist=0.02).fit(data1)
mapper1 = umap.UMAP(densmap=True, dens_lambda=4,
                    n_neighbors=10, min_dist=0.2).fit(data1)

umap.plot.diagnostic(mapper1, diagnostic_type='vq')

"""

"""
# reducer = umap.UMAP(n_neighbors=30, min_dist=0.10)
# embedding = reducer.fit_transform(data)
# import joblib
# save_filename = 'Car25k-reduc1.sav'
# joblib.dump(reducer, save_filename)
# loaded_reducer = joblib.load(save_filename)


"""

"""
# ---- Display interactive plot of data1[]  --------

hover_data1 = pd.DataFrame({'index': np.arange(n_ROI),
                           'label': pname1})
p1 = umap.plot.interactive(mapper1, labels=pname1,
                           hover_data=hover_data1, point_size=3)
umap.plot.show(p1)
"""

"""
# fit = umap.UMAP(n_neighbors=60, min_dist = 0.15)
# u = fit.fit_transform(data)
# plt.scatter(u[:,0], u[:,1], c=data[:,0:4])
# plt.scatter(u[:, 0], u[:, 1])
# plt.title('UMAP embedding of car images')
"""


"""
reducer = umap.UMAP(n_neighbors=30, min_dist=0.10)
embedding = reducer.fit_transform(data)
import joblib
save_filename = 'CarB10kB-reduce.sav'
joblib.dump(reducer, save_filename)


fout = "G2-cars10k.txt"

with open(fout, 'w') as f:
    f.write("idx,x,y,fname\n")
    for j in range(len(data)):
        f.write("%d, %5.3f,%5.3f, %s.jpg\n" % (j, mapper.embedding_[j, 0],
                                        mapper.embedding_[j,1], pname[j]))
f.close()

fout = "G2-cars3636.csv"
with open(fout, 'w') as f:
    f.write("idx,x,y,fname\n")
    for j in range(n_ROI):
        f.write("%d, %5.3f,%5.3f, %s.jpg\n" % (j, mapper1.embedding_[j, 0],
                                        mapper1.embedding_[j,1], pname1[j]))
f.close()

"""


"""
# find data element nearest each to each regularly-spaced grid vertex

p = np.array([1, 1])

xsteps = 10
ysteps = 20
xa = -3    # min..max X range on UMAP axis
xb = 12
ya = -4    # min..max Y range on UMAP axis
yb = 14
xs = (xb - xa) / xsteps
ys = (yb - ya) / ysteps

print("x,y,n,dist,fname")
for x in range(xsteps):
    for y in range(ysteps):
        xp = xa + x*xs
        yp = ya + y*ys
        p = np.array([xp, yp])
        (index, dist) = jmin(p, mapper1.embedding_)
        print("%d,%d,%d,%5.2f,%s" %
              (x, y, index, dist, pname1[index]), end='\n')
"""

# umap.plot.diagnostic(mapper1, diagnostic_type='pca')

# find distance between image 'idx' and all other images in dataset 'g'
# with associated image filenames 'fn'
# sort list to find closest n images, and display
# set g=mapper.embedding_  for example:
#  doGroup(111,mapper.embedding_, pname)

#  doGroup(482,mapper1.embedding_, pname1)
#  482,5.933,3.358,DH5_211111_104513_829_car  (Subaru area)



def doGroup(idx, g, fn):
    print("%d,%5.3f,%5.3f,%s" % (idx, g[idx,0], g[idx,1], fn[idx]))
    elems = len(g)
    darr = np.zeros((elems), dtype=np.float32)  # items,parts of item
    p = g[idx]
    for i in range(elems):
        d = g[i]
        darr[i] = jdist(p, d)  # distance between points

    dists = pd.DataFrame({
        "idx":  range(elems),
        "dist": darr
        })  # create a Pandas data frame

    n = 20  # how many closest images to find
    sd = dists.sort_values(by=['dist'])  # sorted dataframe
    igroup = sd.iloc[0:n, 0].to_numpy()  # indexes of closest points
    # create new dataframe to pass selected items for display
    df = pd.DataFrame(igroup, columns=['index'])
    df.insert(1, 'dist', np.take(darr, igroup))
    df.insert(2, 'fname', np.take(fn, igroup))
    showPlot.p20(df)  # display set of images from fixed data file
    print(df)



"""
# Display a large graph with Datashader to show overlaid points
cardf = pd.DataFrame(mapper.embedding_, columns=['x', 'y'])
cvs = ds.Canvas(plot_width=200, plot_height=200)
agg = cvs.points(cardf, 'x', 'y')
fig = px.imshow(agg)
plot(fig)
"""

"""
# img = ds.tf.shade(agg, cmap=colorcet.kr, how='linear')
# img
points = hv.Points(cardf)
points
hv.output(backend="matplotlib")
hd.datashade(points)
hd.shade(hv.Image(agg))
hv.output(backend="bokeh")
hd.datashade(points)
"""

"""
np.max(mapper1._rhos)  # maximum rho value in dataset
np.max(mapper1._sigmas) # maximum sigma value
"""

"""
xr = [4.05, 19]
yr = [-9.35, 5.925]
j = 0
data1 = np.zeros((n_ROI, px_count), dtype=np.float32)  # ROI only
pname1 = np.ndarray(n_ROI, dt)  # filenames in ROI
inpos = mapper.embedding_
for i in range(len(inpos)):
    p = inpos[i]
    if p[0]>xr[0] and p[0]<xr[1] and p[1]>yr[0] and p[1]<yr[1]:
        print("%d,%d,%5.3f,%5.3f,%s" %(j, i, p[0], p[1], pname[i]))
        data1[j, :] = data[i, :]
        pname1[j] = pname[i]  # save the ROI name from original names
        j += 1
n_ROI = j  # count of data items within ROI
# np.save("data_6413",data1)
# np.save("pname_6413",pname1)
"""
