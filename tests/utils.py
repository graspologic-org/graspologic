# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import os


def data_file(filename):
    return os.path.join(os.path.dirname(__file__), "test_data", filename)
