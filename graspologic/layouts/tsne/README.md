# t-SNE Layouts
This layout uses the node2vec embeddings and t-SNE to reduce the
dimentionality to 2D for the node positions.

To run this layout you can use a command like the following

```
python -m graspologic.layouts.tsne --edge_list edgelist.csv --location_file out/locations-n2vtsne.csv --image_file out/image-n2vtsne.png --perplexity 30 --num_iters 1000
```

# Understanding the t-SNE Derived layouts
The locations of the node clusters are difficult to interpret. For more information on this please see:
https://distill.pub/2016/misread-tsne/
