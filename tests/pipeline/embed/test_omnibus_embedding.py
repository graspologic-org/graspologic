# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from graspologic.pipeline.embed import omnibus_embedding


class TestOmnibusEmbedding(unittest.TestCase):
    def test_raises_not_implemented(self):
        with self.assertRaises(TypeError):
            omnibus_embedding()
