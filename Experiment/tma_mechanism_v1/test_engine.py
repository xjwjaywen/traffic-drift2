"""Synthetic trajectory and component checks; no server data."""
import unittest

import numpy as np
import torch

from engine import fit_variant
from run import PROTOCOL, fit_paper


class EngineTests(unittest.TestCase):
    def test_full_arm_matches_original_care_exactly(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        source = torch.nn.Linear(4, 3)
        tx, rx = torch.randn(13, 4), torch.randn(9, 4)
        ty, ry = np.arange(13) % 3, np.arange(9) % 3
        teacher = source(rx).detach()
        protocol = dict(PROTOCOL, epochs=3, batch_size=8)
        expected, trace = fit_paper(source, tx, ty, rx, ry, teacher,
                                     True, 7, 'cpu', protocol)
        actual, info = fit_variant(source, tx, ty, rx, ry, teacher,
                                   True, True, 7, 'cpu', protocol)
        self.assertEqual(info['optimizer_steps'], trace['optimizer_steps'])
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_four_arms_share_step_count_and_leave_source_unchanged(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        source = torch.nn.Linear(4, 3)
        frozen = {k: v.clone() for k, v in source.state_dict().items()}
        tx, rx = torch.randn(13, 4), torch.randn(9, 4)
        ty, ry = np.arange(13) % 3, np.arange(9) % 3
        teacher = source(rx).detach()
        protocol = dict(PROTOCOL, epochs=2, batch_size=8)
        results = {}
        for replay in (False, True):
            for distill in (False, True):
                head, trace = fit_variant(source, tx, ty, rx, ry, teacher,
                                          replay, distill, 7, 'cpu', protocol)
                results[(replay, distill)] = head
                self.assertEqual(trace['optimizer_steps'], 10)
        self.assertFalse(torch.equal(results[(False, False)].weight,
                                     results[(True, True)].weight))
        for name, value in frozen.items():
            self.assertTrue(torch.equal(source.state_dict()[name], value))


if __name__ == '__main__':
    unittest.main()
