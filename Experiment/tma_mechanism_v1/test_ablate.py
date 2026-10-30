"""Small end-to-end fixture for matched ablation output and resume."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import ablate
from data import finish, sha
from pilot import run_seed
import run as paper


class AblateTests(unittest.TestCase):
    def test_completed_paper_run_is_unchanged_and_full_arm_matches(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            cache = root / 'cache'
            cache.mkdir()
            rng = np.random.default_rng(8)
            roles = ['train', 'val', 'history', 'reference']
            roles += [m + suffix for m in paper.MONTHS for suffix in ('_query', '_eval')]
            for j, role in enumerate(roles):
                y = np.repeat([1, 2], 64)
                if role.endswith(('_query', '_eval')):
                    y = np.r_[y, np.repeat(3, 10)]
                x = rng.normal(y[:, None] * 3, .4, size=(len(y), len(paper.FEATURES))).astype(np.float32)
                ids = np.arange(len(y), dtype=np.uint64) + np.uint64(2**63 + j * 1000)
                np.savez_compressed(cache / (role + '.npz'), x=x, y=y, ids=ids,
                    times=np.full(len(y), pd.Timestamp('2020-01-17', tz='UTC').value))
            finish(cache, dict(synthetic=True), [role + '.npz' for role in roles])
            args = argparse.Namespace(device='cpu', epochs=1, steps=2,
                                      budget=10, rf_trees=2, threads=1)
            run_seed(cache, root / 'seed_0', args, 0)
            protocol = dict(paper.PROTOCOL, epochs=1)
            paper_out = paper.run(root, device='cpu', threads=1, protocol=protocol)
            original = {p: sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(ablate, 'PROTOCOL', protocol):
                output = ablate.run(root, device='cpu', threads=1,
                                    months=['2020-03'], selectors=['badge'])
                table = pd.read_csv(output / 'summary.csv')
                self.assertEqual(len(table), 7)
                part = output / '2020-03/badge'
                with np.load(part / 'predictions.npz', allow_pickle=False) as x:
                    self.assertTrue(np.array_equal(x['paper_care'], x['matched_care']))
                marker = sha(output / 'complete.json')
                ablate.run(root, device='cpu', threads=1,
                           months=['2020-03'], selectors=['badge'])
                self.assertEqual(sha(output / 'complete.json'), marker)
            for path, digest in original.items():
                self.assertEqual(sha(path), digest, str(path))
            self.assertTrue((paper_out / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
