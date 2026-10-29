#!/usr/bin/env python3
"""Server-only TMA acquisition and streaming data audit; no training or tuning."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time
from urllib.parse import urlparse

FILES = {
    'features_nov-apr.tar.gz': '1b_Ynmnjd7a0xeCcYxY8Ov-MZkQvxcaEo',
    'devices.txt': '1zKOTMgufL9CmTaudLd0C0wrUHdh9TBn2',
}
FEATURES = ['srcPort', 'destPort', 'bytes_out', 'num_pkts_out', 'bytes_in',
            'num_pkts_in', 'f_ipt_mean', 'f_ipt_std', 'f_ipt_var', 'f_ipt_skew',
            'f_ipt_kurtosis', 'f_b_mean', 'f_b_std', 'f_b_var', 'f_b_skew',
            'f_b_kurtosis', 'duration', 'pr', 'domainId']

class ForwardReader(io.RawIOBase):
    """tar stream members cannot seek; expose a truthful IO interface to pandas."""
    def __init__(self, source): self.source = source
    def readable(self): return True
    def seekable(self): return False
    def readinto(self, buffer):
        chunk = self.source.read(len(buffer))
        buffer[:len(chunk)] = chunk
        return len(chunk)

def write_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}; self.inside = False
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.action = a.get('action'); self.inside = True
        if tag == 'input' and self.inside and 'name' in a:
            self.fields[a['name']] = a.get('value', '')
    def handle_endtag(self, tag):
        if tag == 'form': self.inside = False

def preflight(root):
    import requests, numpy, pandas, sklearn, torch
    root.mkdir(parents=True, exist_ok=True)
    info = dict(python=sys.version, executable=sys.executable, host=platform.node(),
                platform=platform.platform(), data_root=str(root),
                free_bytes=shutil.disk_usage(root).free,
                versions={n: importlib.metadata.version(n) for n in
                          ['requests','numpy','pandas','scikit-learn','torch']},
                cuda=torch.cuda.is_available(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    write_json(root/'environment.json', info)
    print(json.dumps(info, indent=2), flush=True)

def download(root):
    import requests
    raw = root/'raw'; raw.mkdir(parents=True, exist_ok=True)
    for name, fid in FILES.items():
        out = raw/name; record = raw/(name+'.manifest.json')
        if out.exists():
            if not record.exists() or sha(out) != json.loads(record.read_text())['sha256']:
                raise RuntimeError('Existing file has no valid manifest: '+str(out))
            print('Verified existing', name, flush=True); continue
        if shutil.disk_usage(raw).free < 4*1024**3:
            raise RuntimeError('Need at least 4 GiB free space before downloading')
        part = out.with_name(out.name+'.part')
        # Partial transfers restart explicitly; never interpret an HTML page as a dataset.
        for attempt in range(3):
            try:
                with requests.Session() as s:
                    r = s.get('https://drive.usercontent.google.com/download',
                              params={'id':fid,'export':'download'}, stream=True, timeout=(30,120))
                    r.raise_for_status()
                    if 'text/html' in r.headers.get('Content-Type',''):
                        form = DownloadForm(); form.feed(r.text); r.close()
                        if not form.action or urlparse(form.action).hostname != 'drive.usercontent.google.com':
                            raise RuntimeError('Google Drive did not return the public download form')
                        r = s.get(form.action, params=form.fields, stream=True, timeout=(30,120))
                        r.raise_for_status()
                    if 'text/html' in r.headers.get('Content-Type',''):
                        raise RuntimeError('Google Drive returned HTML (quota/login/error); no data accepted')
                    n=0; last=time.monotonic(); digest=hashlib.sha256()
                    with part.open('wb') as f:
                        for b in r.iter_content(1024*1024):
                            f.write(b); digest.update(b); n+=len(b)
                            if time.monotonic()-last > 20:
                                print(name, f'{n/1024**2:.1f} MiB', flush=True); last=time.monotonic()
                    expected = r.headers.get('Content-Length')
                    if expected and not r.headers.get('Content-Encoding') and n != int(expected):
                        raise RuntimeError('Incomplete HTTP body')
                    if n == 0: raise RuntimeError('Empty download')
                    if name.endswith('.gz'):
                        with gzip.open(part,'rb') as f:
                            while f.read(4*1024*1024): pass
                    else:
                        text = part.read_text()
                        if '<html' in text.lower(): raise RuntimeError('HTML instead of device map')
                    part.replace(out)
                    write_json(record, dict(file=name, google_drive_id=fid, bytes=n,
                        sha256=digest.hexdigest(), source_repository='https://github.com/DADABox/revisiting-iot-device-identification',
                        checksum_note='Locally computed integrity hash; publisher checksum unavailable'))
                    print('Completed',name,n,flush=True)
                    break
            except Exception as e:
                print('Attempt',attempt+1,'failed:',str(e),flush=True)
                if attempt == 2: raise
                time.sleep(3)

def audit_archive(archive, output):
    import numpy as np
    import pandas as pd
    output.mkdir(parents=True,exist_ok=True)
    completed=output/'audit_complete.json'
    if completed.exists():
        old=json.loads(completed.read_text())
        if old['archive_sha256'] != sha(archive): raise RuntimeError('Archive changed')
        if old['script_sha256'] != sha(Path(__file__)): raise RuntimeError('Audit code changed; use new output directory')
        if any(sha(output/n)!=h for n,h in old['artifacts'].items()): raise RuntimeError('Audit artifact changed')
        print('Audit already complete and verified'); return
    monthly=Counter(); weekly=Counter(); protocols=Counter(); invalid=Counter(); nonfinite=Counter()
    members=[]; rows=0; first=None; last=None; available=None
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile(): continue
            members.append(dict(name=member.name,bytes=member.size))
            if not member.name.lower().endswith('.csv'): continue
            # Stream directly without extracting archive paths or executable content.
            stream=io.BufferedReader(ForwardReader(tar.extractfile(member)))
            for df in pd.read_csv(stream,chunksize=100000,low_memory=False):
                if not {'time_start','deviceId'}.issubset(df.columns):
                    raise RuntimeError('Unexpected schema in '+member.name+': '+repr(list(df.columns)))
                if available is None: available=list(df.columns)
                if available != list(df.columns): raise RuntimeError('CSV schemas differ')
                rows+=len(df)
                dt=pd.to_datetime(df.time_start,errors='coerce',utc=True)
                label=pd.to_numeric(df.deviceId,errors='coerce')
                valid_label=label.notna() & np.isfinite(label) & (label % 1 == 0) & label.between(1,41)
                valid=dt.notna() & valid_label
                invalid['time']+=int(dt.isna().sum()); invalid['label']+=int((~valid_label).sum())
                invalid['either']+=int((~valid).sum())
                d=dt[valid]; y=label[valid].astype(int)
                if len(d):
                    lo=d.min().isoformat(); hi=d.max().isoformat()
                    first=min(first,lo) if first else lo;last=max(last,hi) if last else hi
                    monthly.update(zip(d.dt.strftime('%Y-%m'),y))
                    weekly.update(zip(d.dt.strftime('%G-W%V'),y))
                for col in FEATURES:
                    if col in df:
                        a=pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)
                        nonfinite[col]+=int((~np.isfinite(a)).sum())
                if 'pr' in df: protocols.update(df.pr.fillna('MISSING').astype(str))
                print('Audited',rows,'rows',flush=True)
    if not rows: raise RuntimeError('No CSV records found')
    def counts(name, counter, key):
        with (output/name).open('w',newline='') as f:
            w=csv.writer(f); w.writerow([key,'device_id','rows'])
            w.writerows((period,c,n) for (period,c),n in sorted(counter.items()))
    counts('monthly_class_counts.csv',monthly,'month');counts('weekly_class_counts.csv',weekly,'week')
    report=dict(raw_rows=rows,valid_time_label_rows=rows-invalid['either'],
        first_timestamp_utc=first,last_timestamp_utc=last,
        observed_device_ids=sorted({c for _,c in monthly}),
        invalid_rows=dict(invalid),nonfinite_feature_counts=dict(nonfinite),
        protocol_counts=dict(protocols),columns=available,archive_members=members,
        missing_expected_features=sorted(set(FEATURES)-set(available or [])),
        units='Author-provided TCP/UDP flow records; encryption not established by filename',
        limitations=['No deduplication performed; duplicate/session leakage audit needed before training.',
          'Time coverage and count shifts do not establish classifier degradation or absorber collapse.',
          'Device IDs index this testbed; physical device identity is not cross-device generalization.',
          'No training, parameter selection, target repair or future-label querying performed.'])
    write_json(output/'data_audit.json',report)
    (output/'REPORT.txt').write_text(
        'TMA 数据检查完成（未训练模型）\n'+
        f'原始记录：{rows:,}；时间与标签有效：{rows-invalid["either"]:,}\n'+
        f'时间：{first} 至 {last}\n设备 ID 数：{len(report["observed_device_ids"])}\n'+
        '请返回 data_audit.json、monthly_class_counts.csv 和 weekly_class_counts.csv。\n'+
        '这些结果只能确认数据结构与覆盖，不能宣称存在类别崩溃。\n')
    names=['data_audit.json','monthly_class_counts.csv','weekly_class_counts.csv','REPORT.txt']
    write_json(completed,dict(archive_sha256=sha(archive),script_sha256=sha(Path(__file__)),
                             artifacts={n:sha(output/n) for n in names}))
    print((output/'REPORT.txt').read_text(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','download','audit','all'])
    p.add_argument('--data-root',type=Path,default=Path('/data/xjw/tma-care-data'))
    a=p.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Run this entry on the Linux server only. No local dataset download is permitted.')
    root=a.data_root.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.stage0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command in ('preflight','all'):preflight(root)
        if a.command in ('download','all'):download(root)
        if a.command in ('audit','all'):
            archive=root/'raw/features_nov-apr.tar.gz'
            m=archive.with_name(archive.name+'.manifest.json')
            if not m.exists() or sha(archive)!=json.loads(m.read_text())['sha256']:
                raise RuntimeError('Run verified download first')
            audit_archive(archive,root/'audit_v1')

if __name__=='__main__':main()
