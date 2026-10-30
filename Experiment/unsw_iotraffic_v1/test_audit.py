"""Synthetic ZIP coverage check; no external dataset needed."""
from pathlib import Path
import tempfile
import unittest
import zipfile

import pandas as pd

from audit import audit, write_report


class AuditTests(unittest.TestCase):
    def test_month_counts_and_invalid_times(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'flows.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for device, times in [('A', ['2024-01-01', '2024-02-01', 'bad']),
                                      ('B', ['2024-02-03', '2024-02-04'])]:
                    stamp = pd.to_datetime(times, utc=True, errors='coerce')
                    frame = pd.DataFrame(dict(time=[v.timestamp() if not pd.isna(v)
                                                    else 'bad' for v in stamp],
                                              ipProto=[6] * len(times),
                                              srcNumPackets=[1] * len(times)))
                    archive.writestr(f'flows/{device}_flows.csv', frame.to_csv(index=False))
            devices, months, columns = audit(path, chunksize=2)
            self.assertEqual(devices.set_index('device').loc['A', 'invalid_time_rows'], 1)
            table = months.set_index(['device', 'month']).rows
            self.assertEqual(table.loc[('A', '2024-01')], 1)
            self.assertEqual(table.loc[('A', '2024-02')], 1)
            self.assertEqual(table.loc[('B', '2024-02')], 2)
            output = write_report(path, root / 'audit', devices, months, columns)
            self.assertTrue((output / 'manifest.json').exists())
            with self.assertRaisesRegex(ValueError, 'new and empty'):
                write_report(path, output, devices, months, columns)


if __name__ == '__main__':
    unittest.main()
