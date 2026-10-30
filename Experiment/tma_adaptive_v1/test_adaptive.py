import unittest

import numpy as np
import torch

from adaptive import drift_flags, fit_adaptive
from run import PROTOCOL, fit_paper


class AdaptiveTests(unittest.TestCase):
    def test_flags_use_paid_true_class_errors_and_historical_eligibility(self):
        historical = dict(support=np.array([100, 100, 20]),
                          recall=np.array([.9, .9, 1.]))
        labels = np.array([0] * 6 + [1] * 6 + [2] * 6 + [-1] * 3)
        source = np.array([1] * 6 + [1] * 6 + [0] * 6 + [0] * 3)
        flags, rows = drift_flags(labels, source, historical, 3)
        self.assertEqual(flags.tolist(), [True, False, False])
        self.assertEqual(rows[0]['paid_true_class_support'], 6)
        self.assertEqual(rows[0]['source_errors'], 6)

    def test_no_flags_exactly_reproduces_original_care(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        source = torch.nn.Linear(4, 3)
        tx, rx = torch.randn(13, 4), torch.randn(9, 4)
        ty, ry = np.arange(13) % 3, np.arange(9) % 3
        teacher = source(rx).detach()
        protocol = dict(PROTOCOL, epochs=3, batch_size=8)
        expected, trace = fit_paper(source, tx, ty, rx, ry, teacher,
                                     True, 7, 'cpu', protocol)
        actual, info = fit_adaptive(source, tx, ty, rx, ry, teacher,
                                    [False] * 3, 7, 'cpu', protocol)
        self.assertEqual(info['optimizer_steps'], trace['optimizer_steps'])
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_relaxation_removes_old_class_reference_ce_and_keeps_step_count(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        source = torch.nn.Linear(4, 3)
        tx, rx = torch.randn(13, 4), torch.randn(9, 4)
        ty, ry = np.arange(13) % 3, np.arange(9) % 3
        teacher = source(rx).detach()
        protocol = dict(PROTOCOL, epochs=3, batch_size=8)
        full, full_trace = fit_adaptive(source, tx, ty, rx, ry, teacher,
                                        [False] * 3, 7, 'cpu', protocol)
        adapted, trace = fit_adaptive(source, tx, ty, rx, ry, teacher,
                                      [True, False, False], 7, 'cpu', protocol)
        self.assertEqual(trace['optimizer_steps'], full_trace['optimizer_steps'])
        self.assertEqual(trace['reference_ce_rows'], 6)
        self.assertFalse(torch.equal(full.weight, adapted.weight))


if __name__ == '__main__':
    unittest.main()
