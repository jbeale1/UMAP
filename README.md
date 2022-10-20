# UMAP
## Finding patterns with no prior knowledge

UMAP is described in this [readthedocs](https://umap-learn.readthedocs.io/en/latest/parameters.html) page.

A sample data set "cars-7390.csv" records the velocity and size of traffic detected passing by on the road.
Here is a small [sample](https://github.com/jbeale1/UMAP/blob/main/cars_sample.csv) of that data.  "v1,v2,v3" are average velocity of the beginning, middle and end parts of the motion of the traffic as it passes by the camera. "std" is the standard deviation of velocity throughout the event. "minY" is the lowest vertical position motion is found in the frame. "pixels" is proportional to the apparent size of the moving object.

UMAP finds clusters in this data, without any training, supervision, labelling or a-priori knowlege. The code that generated the two output images below is in straight python
[here](https://github.com/jbeale1/UMAP/blob/main/umap-example1.py), and as a jupyter notebook [here](https://github.com/jbeale1/UMAP/blob/main/umap-example1.ipynb).

Upon inspection, the clusters UMAP finds turn out to correspond to specific kinds of traffic. As far as I can tell, _all_ the large variety of structure visible in the output map is showing real features of the data, although some parts of it are easier to interpret than others.

As a concrete example, the mail truck comes by once a day, more or less. The regular mailman gets mapped to the same spot each time, because he consistently drives the same way. When we had a new driver who drove differently, her data showed up in a very different place on the map (third line of data below). This wasn't due to the date and time; I tried changing that and it didn't affect the output. The majority of traffic does not slow down near the mailbox, so all those points get mapped elsewhere.
```
day,hour,minute,second,frames,v1,v2,v3,std,minY,size => [ output mapping ]
14,14,42,58.500,59,2.73,2.32,1.90,2.53,072,5425  => [5.1577015, 1.5967405]  old mailman
15,13,39,30.980,51,3.43,2.85,2.17,2.52,075,5875  => [5.166201 , 1.6030715]  old mailman
19,17,31,11.982,51,3.39,2.77,2.08,1.50,076,6220  => [8.317883 , 7.9654984]  new person
```

This output map is colored by the average velocity of the detected object. Slower traffic is near the middle, faster towards the edges. Traffic moving to the right is toward the right-hand side of the graph.
![UMAP plot colored by speed](https://github.com/jbeale1/UMAP/blob/main/pics/car-data-Oct17-ColorV2-annotated.png?raw=true)

This output map is colored by the size of the detected object. Larger traffic (eg. school bus, garbage truck) are towards the top, and smaller (bicycle, person) are towards the bottom.
![UMAP plot colored by size](https://github.com/jbeale1/UMAP/blob/main/pics/car-data-Oct17-ColorSize-annotated.png?raw=true)

Animation comparing the training data mapping, and later-acquired test data that fits well into the existing clusters.
![comparision of train and test mappings](https://github.com/jbeale1/UMAP/blob/main/pics/CompareTrainTest.gif?raw=true)

Pair Plot of the more significant variables against each other (using training data).
![pair plot](https://github.com/jbeale1/UMAP/blob/main/pics/cars-pair-plot.png?raw=true)

