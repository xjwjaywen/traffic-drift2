"""Artificial fixtures only. Never downloads data or runs training."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import stage0

class AuditTests(unittest.TestCase):
    def test_counts_integrity_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'fixture.tar.gz'
            body=b'time_start,deviceId,bytes_out,pr\n2019-11-01,1,100,6\n2020-01-01,2,inf,17\nBAD,2,5,6\n2020-01-02,99,2,6\n'
            with tarfile.open(archive,'w:gz') as t:
                m=tarfile.TarInfo('fixture.csv');m.size=len(body);t.addfile(m,io.BytesIO(body))
            stage0.audit_archive(archive,root/'out')
            report=json.loads((root/'out/data_audit.json').read_text())
            self.assertEqual(report['raw_rows'],4)
            self.assertEqual(report['valid_time_label_rows'],2)
            self.assertEqual(report['observed_device_ids'],[1,2])
            self.assertEqual(report['invalid_rows'],{'time':1,'label':1,'either':2})
            self.assertEqual(report['nonfinite_feature_counts']['bytes_out'],1)
            stage0.audit_archive(archive,root/'out')
            (root/'out/REPORT.txt').write_text('corrupt')
            with self.assertRaisesRegex(RuntimeError,'artifact changed'):
                stage0.audit_archive(archive,root/'out')
    def test_public_download_form(self):
        f=stage0.DownloadForm()
        f.feed('<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="id" value="x"><input name="confirm" value="t"></form>')
        self.assertEqual(f.fields,{'id':'x','confirm':'t'})

if __name__=='__main__':unittest.main()
