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

class Response:
    def __init__(self,status,headers,chunks=(),text=''):
        self.status_code=status;self.headers=headers;self.chunks=chunks;self.text=text
    def raise_for_status(self):
        if self.status_code>=400:raise RuntimeError('HTTP error')
    def iter_content(self,n):
        for item in self.chunks:
            if isinstance(item,Exception):raise item
            yield item
    def close(self):pass

class Session:
    def __init__(self,*responses):self.responses=list(responses);self.calls=[]
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return self.responses.pop(0)

class ResumeTests(unittest.TestCase):
    def test_timeout_then_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file.part'
            session=Session(Response(200,{'Content-Length':'6','ETag':'"v1"'},[b'abc',TimeoutError('timeout')]))
            with self.assertRaises(TimeoutError):stage0.transfer(session,'id',p)
            self.assertEqual(p.read_bytes(),b'abc')
            session=Session(Response(206,{'Content-Range':'bytes 3-5/6','ETag':'"v1"'},[b'def']))
            self.assertEqual(stage0.transfer(session,'id',p),6)
            self.assertEqual(p.read_bytes(),b'abcdef')
            self.assertEqual(session.calls[0][1]['headers']['Range'],'bytes=3-')
            self.assertEqual(session.calls[0][1]['headers']['If-Range'],'"v1"')
    def test_legacy_partial_and_ignored_range(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file.part';p.write_bytes(b'abc')
            session=Session(Response(206,{'Content-Range':'bytes 3-5/6'},[b'def']))
            stage0.transfer(session,'id',p);self.assertEqual(p.read_bytes(),b'abcdef')
            session=Session(Response(200,{'Content-Length':'3'},[b'NEW']))
            stage0.transfer(session,'id',p);self.assertEqual(p.read_bytes(),b'NEW')
    def test_wrong_offset_and_changed_validator(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file.part';p.write_bytes(b'abc')
            session=Session(Response(206,{'Content-Range':'bytes 2-5/6'},[b'cdef']))
            with self.assertRaisesRegex(RuntimeError,'offset'):stage0.transfer(session,'id',p)
            self.assertEqual(p.read_bytes(),b'abc')
            stage0.write_json(p.with_name(p.name+'.http.json'),dict(google_drive_id='id',total=6,validator='old'))
            session=Session(Response(206,{'Content-Range':'bytes 3-5/6','ETag':'new'},[b'def']))
            with self.assertRaisesRegex(RuntimeError,'validator'):stage0.transfer(session,'id',p)
            self.assertEqual(p.read_bytes(),b'abc')
    def test_confirmation_keeps_range(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file.part';p.write_bytes(b'abc')
            html='<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="confirm" value="t"></form>'
            session=Session(Response(200,{'Content-Type':'text/html'},text=html),Response(206,{'Content-Range':'bytes 3-5/6'},[b'def']))
            stage0.transfer(session,'id',p)
            self.assertEqual(session.calls[1][1]['headers']['Range'],'bytes=3-')
            self.assertEqual(p.read_bytes(),b'abcdef')
    def test_complete_partial_416_and_short_body(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file.part';p.write_bytes(b'abc')
            self.assertEqual(stage0.transfer(Session(Response(416,{'Content-Range':'bytes */3'})),'id',p),3)
            with self.assertRaisesRegex(RuntimeError,'inconsistent'):
                stage0.transfer(Session(Response(416,{'Content-Range':'bytes */2'})),'id',p)
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                stage0.transfer(Session(Response(206,{'Content-Range':'bytes 3-5/6'},[b'd'])),'id',p)
            self.assertEqual(p.read_bytes(),b'abcd')

if __name__=='__main__':unittest.main()
