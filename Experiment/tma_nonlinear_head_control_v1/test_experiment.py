import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import fit_rf, paid_queries


class FakeReader:
    def __init__(self, root):
        self.root = root

    def file(self, name):
        return self.root / name


class NonlinearControlTests(unittest.TestCase):
    def test_only_paid_query_labels_are_used(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / '2020-03').mkdir()
            pd.DataFrame(dict(record_id=['10', '12', '13'], device_id=[1, 2, 3])).to_csv(
                root / '2020-03/hybrid_queries.csv', index=False)
            query = dict(ids=np.array([10, 11, 12, 13]), y=np.array([1, 9, 2, 3]))
            selected, labels = paid_queries(FakeReader(root), '2020-03', query, [1, 2], 3)
            np.testing.assert_array_equal(selected, [0, 2, 3])
            np.testing.assert_array_equal(labels, [0, 1, -1])

    def test_forest_can_fit_all_classes_with_references(self):
        rng = np.random.default_rng(9)
        target_x = np.r_[rng.normal(-2, .1, (12, 3)), rng.normal(2, .1, (12, 3))]
        target_y = np.array([0] * 12 + [1] * 12)
        ref_x = np.r_[rng.normal(-2, .1, (5, 3)), rng.normal(2, .1, (5, 3))]
        ref_y = np.array([0] * 5 + [1] * 5)
        model, known = fit_rf(target_x, target_y, ref_x, ref_y, 0, 1)
        self.assertEqual(known, 24)
        np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertGreater((model.predict(target_x) == target_y).mean(), .9)


if __name__ == '__main__':
    unittest.main()
