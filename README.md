# UMAP
## Finding patterns with no prior knowledge

UMAP is described in this [readthedocs](https://umap-learn.readthedocs.io/en/latest/parameters.html) page.

A sample data set "cars-7390.csv" records the velocity and size of traffic detected passing by on the road.
Here is a small [sample](https://github.com/jbeale1/UMAP/blob/main/data/cars_sample.csv) of that data.  "v1,v2,v3" are average velocity of the beginning, middle and end parts of the motion of the traffic as it passes by the camera. "std" is the standard deviation of velocity throughout the event. "minY" is the lowest vertical position motion is found in the frame. "pixels" is proportional to the apparent size of the moving object in the video frame.

UMAP finds clusters of similar values in this data, without any training, supervision, labelling or a-priori knowlege. The code that generated the two output images below is in straight python
[here](https://github.com/jbeale1/UMAP/blob/main/python/umap-example1.py), and as a jupyter notebook [here](https://github.com/jbeale1/UMAP/blob/main/umap-example1.ipynb).

Upon inspection, the clusters UMAP finds turn out to correspond to specific kinds of traffic. As far as I can tell, _all_ the various structure visible in the output map is showing real features of the data, although some parts of it are easier to interpret than others.

As a concrete example, the mail truck comes by once a day, more or less. The regular mailman gets mapped to the same spot in the output map each time, because he consistently drives the same way. When we had a new driver who drove differently, her data showed up in a very different place on the map (third line of data below). This wasn't due to the date and time, which UMAP doesn't see, it is only fed the column 'frames' onwards. The majority of traffic does not slow down near the mailbox, so most cars passing by are mapped elsewhere.
```
day,hour,min,sec,frames,v1,v2,v3,stdv,minY,size  => [ output mapping ]
14,14,42,58.500,59,2.73,2.32,1.90,2.53,072,5425  => [5.1577015, 1.5967405]  old mailman
15,13,39,30.980,51,3.43,2.85,2.17,2.52,075,5875  => [5.166201 , 1.6030715]  old mailman
19,17,31,11.982,51,3.39,2.77,2.08,1.50,076,6220  => [8.317883 , 7.9654984]  new person
```

This output map is colored by the average velocity of the detected object. Slower traffic is near the middle, faster towards the edges. Traffic moving to the right is toward the right-hand side of the graph.
![UMAP plot colored by speed](https://github.com/jbeale1/UMAP/blob/main/pics/car-data-Oct17-ColorV2-annotated.png?raw=true)

This output map is colored by the size of the detected object. Larger traffic (eg. school bus, garbage truck) are towards the top, and smaller (bicycle, person) are towards the bottom.
![UMAP plot colored by size](https://github.com/jbeale1/UMAP/blob/main/pics/car-data-Oct17-ColorSize-annotated.png?raw=true)

Reality check: this animation cycles between 15,000 records of real data recorded in 2021, and the same number of records of random data in which each column separately has the same mean and standard deviation as the corresponding measured data column, and is plotted in the same mapping. In both plots, points are colored by "v2" (velocity). You can see UMAP puts even random data into the same general shapes. However, the differences between these two plots are a fingerprint of the real dataset, due to whatever correlations exist between variables. 
![real data and random](https://github.com/jbeale1/UMAP/blob/main/pics/Car-2021-random-compare.gif?raw=true)

After creating the map with some training data, I put a set of test data into the same map, took the closest
point in the original dataset to each point in the test data, and compared the differences between each variable
in those two points as a function of the distribution over the full dataset.  As hoped, the new points on average, mapped
to the original data set points more closely than half the distribution (0.5) away, in every case. Doing the same test with a random dataset (but having the correct mean and std.dev on each axis separately) showed in that case, the nearest mapped point was essentially uncorrelated (0.5). 
```
frames,  v1,    v2,     v3,   stdev,  minY, size,   dir    
[+0.243 +0.192 +0.146 +0.177 +0.217 +0.303 +0.269 +0.056]    real data
[+0.517 +0.650 +0.647 +0.642 +0.527 +0.573 +0.519 +0.752]    randomized data
```
