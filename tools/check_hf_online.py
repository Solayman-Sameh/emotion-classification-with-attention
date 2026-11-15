"""Verify public HF revision/manifest; optionally test a genuinely empty cache.

Run outside a network-restricted environment. --fresh-cache downloads ~274 MB
into .release-checks/fresh-hf and deliberately does not reuse the normal cache.
"""
import argparse
import json
import pathlib
import sys
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'application'))
from download_models import DEFAULT_HF_REPO_ID, DEFAULT_HF_REVISION, download_models, _manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-cache',action='store_true')
    args=parser.parse_args()
    url='https://huggingface.co/api/models/'+DEFAULT_HF_REPO_ID+'/revision/'+DEFAULT_HF_REVISION+'?blobs=true'
    with urllib.request.urlopen(url,timeout=20) as response:info=json.load(response)
    assert info['sha']==DEFAULT_HF_REVISION
    assert not info.get('private',False),'Expected public model access for the documented anonymous quickstart'
    if args.fresh_cache:
        cache=ROOT/'.release-checks/fresh-hf'
        if cache.exists() and any(cache.iterdir()):
            raise RuntimeError('Fresh-cache test requires an empty cache; use a new verification directory')
        cache.mkdir(parents=True,exist_ok=True)
        snapshot=download_models(cache_dir=cache)
        files=len(_manifest(snapshot/'files_manifest.json'))
    else:
        url='https://huggingface.co/'+DEFAULT_HF_REPO_ID+'/resolve/'+DEFAULT_HF_REVISION+'/files_manifest.json'
        with urllib.request.urlopen(url,timeout=20) as response:manifest=json.load(response)
        files=len(manifest)
        remote={entry['rfilename']:entry for entry in info['siblings']}
        for name,expected in manifest.items():
            assert name in remote,name
            lfs=remote[name].get('lfs') or {}
            if lfs:assert lfs['sha256']==expected,name
    print(json.dumps({'status':'passed','repo':DEFAULT_HF_REPO_ID,'revision':DEFAULT_HF_REVISION,
        'public_access':True,'manifest_files':files,'fresh_cache':args.fresh_cache}))


if __name__=='__main__':main()
