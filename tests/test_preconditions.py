# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from graspologic import preconditions


class TestPreconditions(unittest.TestCase):
    def test_check_arguments(self):
        test_true_expressions = [1 < 3, 3 == 3, True, 1 == 1]
        test_false_expressions = [
            3 < 1,
            3 != 3,
            None is True,
            1 == "1",
        ]
        for resolved_expression in test_true_expressions:
            preconditions.check_argument(resolved_expression, "This should be true")

        for resolved_expression in test_false_expressions:
            with self.assertRaises(ValueError):
                preconditions.check_argument(resolved_expression, "This is false")

    def test_check_argument_types(self):
        preconditions.check_argument_types(1, int, "Some message")
        with self.assertRaises(TypeError):
            preconditions.check_argument_types(1, set, "This fails")

    def test_check_optional_argument_types(self):
        preconditions.check_optional_argument_types(1, int, "Some message")
        preconditions.check_optional_argument_types(None, int, "Some message")
        with self.assertRaises(TypeError):
            preconditions.check_optional_argument_types(1, set, "This fails")
