CLI
===

In addition to the main library, there is also a CLI runnable module for automatically
generating layouts for graphs in an edge list.

You can run this from the command line like so:

.. code-block:: bash

    python -m graspologic.layouts --help

Which should return something like:

.. code-block:: none

    usage: python -m graspologic.layouts [-h] [--verbose VERBOSE] {n2vumap,n2vtsne,render} ...

    Runnable module that automatically generates a layout of a graph by a provided edge list

    positional arguments:
      {n2vumap,n2vtsne,render}
        n2vumap             Auto layout using UMAP for dimensionality reduction
        n2vtsne             Auto layout using tSNE for dimensionality reduction
        render              Renders a graph via an input file

Of those commands, you can then do:

.. code-block:: bash

    python -m graspologic.layouts n2vumap --help

Which will return something like:

.. code-block:: none

    usage: python -m graspologic.layouts n2vumap [-h] --edge_list EDGE_LIST [--skip_header] [--image_file IMAGE_FILE] [--location_file LOCATION_FILE] [--max_edges MAX_EDGES] [--dpi DPI]
                                             [--allow_overlaps]

    optional arguments:
      -h, --help            show this help message and exit
      --edge_list EDGE_LIST
                            edge list in csv file. must be source,target,weight.
      --skip_header         skip first line in csv file, corresponding to header.
      --image_file IMAGE_FILE
                            output path and filename for generated image file. required if --location_file is omitted.
      --location_file LOCATION_FILE
                            output path and filename for location file. required if --image_file is omitted.
      --max_edges MAX_EDGES
                            maximum edges to keep during embedding. edges with low weights will be pruned to keep at most this many edges
      --dpi DPI             used with --image_file to render an image at this dpi
      --allow_overlaps      skip the no overlap algorithm and let nodes stack as per the results of the down projection algorithm
