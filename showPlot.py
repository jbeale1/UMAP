# -*- coding: utf-8 -*-
# Convert input images to small, normalized, grescale version
# J.Beale  12-Dec-2021

import os  # loop over images in directory
import matplotlib.pyplot as plt
import skimage.io   # to read & save images
import numpy as np  # for np.zeros()
# from skimage import exposure  # adaptive hist. equalization
from skimage.transform import resize
from skimage.util import img_as_ubyte
# from skimage.color import rgb2gray
# import pandas as pd    # pd.read_csv()
# --------------------------------------------------------------

xsize = 300  # image size, pixels across
ysize = 115  # image size, pixels down

"""
def inorm(img):
    target_dim = (ysize, xsize)  # convert image to this size bitmap
    gray = rgb2gray(img)  # convert RGB image to greyscale
    img_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    img2 = resize(img_eq, target_dim, anti_aliasing=True)
    return(img_as_ubyte(img2))
"""


def jresize(img):
    img2 = resize(img, (ysize, xsize), anti_aliasing=True)
    return(img_as_ubyte(img2))


def p20(df):
    path = "C:\\Users\\beale\\Documents\\YOLO\\out\\car"  # read images here
    # fin = "Select20-carsB.csv"

    xmul = 4   # how many images along this axis
    ymul = 5

    canvas = np.zeros((ysize*ymul, xsize*xmul, 3), dtype=np.uint8)

    xp = 0  # position of sub-image on canvas
    yp = 0  # position of sub-image on canvas

    # df = pd.read_csv(fin)  # load dataframe with image filenames

    for i in range(xmul * ymul):
        iname = df.loc[i, 'fname']
        xp = int((i % xmul)) * xsize
        yp = int(i / xmul) * ysize
        fname_in = os.path.join(path, iname) + ".jpg"
        # print(fname_in)
        img = skimage.io.imread(fname=fname_in)  # color image input
        img_new = jresize(img)
        canvas[yp:yp+ysize, xp:xp+xsize] = img_new

    fix, ax = plt.subplots()
    ax.imshow(canvas)
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    p20()


"""
fname_out = fin + ".png"
skimage.io.imsave(fname_out, canvas)  # save file
"""
# ------------------------------------
