# UMAP
Experiments with UMAP : finding patterns in data sets, without knowing anything.

UMAP is described in this [readthedocs](https://umap-learn.readthedocs.io/en/latest/parameters.html) page.

A sample data set [cars-7390.csv](https://github.com/jbeale1/UMAP/blob/main/cars-7390.csv) describes the velocity and size of traffic passing by on the road.
UMAP finds clusters in this data, without any training. Upon inspection, these clusters turn out to correspond to specific kinds of traffic.

This output map is colored by the average velocity of the detected object. Slower traffic is near the middle, faster towards the edges. Traffic moving to the right is toward the right-hand side of the graph.
![UMAP plot colored by speed](https://github.com/jbeale1/UMAP/blob/main/car-data-Oct17-ColorV2-annotated.png?raw=true)

THis output map is colored by the size of the detected object. Larger traffic (eg. school bus, garbage truck) are towards the top, and smaller (bicycle, person) are towards the bottom.
![UMAP plot colored by size](https://github.com/jbeale1/UMAP/blob/main/car-data-Oct17-ColorSize-annotated.png?raw=true)

