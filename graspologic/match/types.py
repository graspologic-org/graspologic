# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union

import numpy as np
from typing_extensions import Literal

from graspologic.types import AdjacencyMatrix, List, Tuple

MultilayerAdjacency = Union[List[AdjacencyMatrix], AdjacencyMatrix, np.ndarray]

PaddingType = Literal["adopted", "naive"]

PartialMatchType = Union[np.ndarray, Tuple]
