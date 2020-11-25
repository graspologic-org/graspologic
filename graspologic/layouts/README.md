# Introduction
This repository combines the Essex autolayout and the node2vec embedding + UMAP layout into one easy to launch location.  It allows you to produce an image file of the resulting layout or a layout CSV file or both. It requires an edge list of the form (Source, Target, weight), and that edge list must include a header.

## Generating Layouts

To run this code execute the following command:
```
python -m graspologic.layouts.umap --edge_list edgelist.csv --location_file out/locations-n2vumap.csv --image_file out/image-n2vumap.png
```
or
```
python -m graspologic.layouts.umap--edge_list edgelist.csv --location_file out/locations-tsne.csv --image_file out/image-tsne.png --layout_type n2vtsne
```
If you want to change the colors you must pass a location file value to the command. That value will be used to re-render the graph.

# To Customize the Graph Output

Another module is provided to customize layout coloring and edges. The following sections apply to this module. The command is:
```
python -m graspologic_layouts.render_only --location_file sample-locations.csv  --image_file sample-render.png
```
The location_file should be the output of the layout process. The customization below can change communities and show edges.

There are a few additional command line arguments that are not needed to get a layout but can affect the rendered image.

## Show Edges

If you want to show the edges you can provide an edge file with the:
```
--edge_list <edge_file>
```
option.

## Dark Background
```
--dark_background
```
This will put the image on a dark background.

## Advanced OPTIONAL Node Metadata coloring

An attribute from a node metadata file can be used for coloring. The default used in the initial layout is the
result of a Leiden partitioning. If a differnt partitioning is desired a CSV file can be specified like this:
```
--node_file sample_attributes.csv
```
The file must contain a header line with column names. One column must be the node identifier and another must be the color attribute. It could like like:
```
nodeid,attribute1,attribute2,attribute3
1,14,soccer,bananna
2,7,ping pong,grapefruit
3,7,badmiton,mango
4,7,badmiton,bananna
...
```
The attribute is assumed to be categorical. For continous values see below.

Two other command line options must be specified, the name of the node identifier attribute, and the name of the attribute to use for coloring.
It could look like this:
```
--node_id nodeid --color_attribute attribute3
```

### Continous Valued Coloring
If you have a continuous attribute that you would like to map onto the color field you can add the flag color_is_continuous. With this flag
the range of the graph is linearly scaled onto a continout color range.
```
--color_is_continuous
```
This changes the color_scheme from nominal to sequential.

The default color file provides the following values for categorical partitioning to the color_scheme: 'nominal', 'nominalBold', 'nominalMuted'
It also provides the following values for continous color values: 'sequential', 'sequential2', 'diverging', and 'diverging2'
They can be added to the command line. The default for categorical values is nominal and the default for continuous is sequential.
```
--color_scheme diverging
```

## Changing the colors
A default color scheme is provided. If you prefer a different color scheme you can create one yourself from the thematic website:
https://microsoft.github.io/thematic/ Select an accent color and I recommend changing the scale item to 100, then choose the JSON
format and download the file.  That is the format that the renderer is expected. You can then specify this file on the command line
with the color_file command line arguement.
```
--color_file new_color_file.json
```

