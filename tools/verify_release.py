"""Verify staged release bytes and frozen notebooks independently of app revisions."""
import ast
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
def digest(data): return hashlib.sha256(data).hexdigest()
def git(*args):
    return subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,check=True).stdout


def main():
    app = ROOT/'application'
    manifest = json.loads((app/'files_manifest.json').read_text(encoding='utf-8'))
    missing = []
    checked = 0
    for name, expected in manifest.items():
        file = app/name
        if not file.is_file():
            missing.append(name)
            continue
        assert digest(file.read_bytes()) == expected, 'Working file mismatch: '+name
        staged = git('cat-file','--filters','--path=application/'+name,':application/'+name)
        assert digest(staged) == expected, 'Checkout byte mismatch: '+name
        checked += 1
    assert set(missing) <= {'models/distilbert/model.safetensors','models/bilstm_attention/best.pt'}, missing
    notebooks = {}
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        values = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            for node in ast.parse(''.join(cell['source'])).body:
                if (isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                        and node.targets[0].id in ['EMBEDDED_FILES','EMBEDDED_SHA256']):
                    values[node.targets[0].id] = ast.literal_eval(node.value)
        assert values.get('EMBEDDED_FILES'), 'Missing embedded runtime: '+path.name
        hashes = {name:digest(text.encode()) for name,text in values['EMBEDDED_FILES'].items()}
        assert hashes == values['EMBEDDED_SHA256'] == notebook['metadata']['emotion_project']['embedded_sha256'], path.name
        relative = path.relative_to(ROOT).as_posix()
        assert git('show', ':'+relative).decode('utf-8') == path.read_text(encoding='utf-8'), 'Unstaged notebook changes: '+path.name
        notebooks[path.name] = len(hashes)
    assert (ROOT/'requirements.txt').read_text().strip() == '-r application/requirements.txt'
    paths = [p for p in git('ls-files','-z').decode().split('\0') if p]
    assert not any(p.endswith(('.pt','.safetensors','.zip','.pyc')) for p in paths)
    assert not any('/.release-checks/' in '/'+p for p in paths)
    assert not any(any(part in {'.venv','__pycache__','.ipynb_checkpoints','emotion_classification_run','codex-space'}
                       for part in pathlib.PurePosixPath(p).parts) for p in paths)
    assert not any(pathlib.PurePosixPath(p).name.startswith('.env') and pathlib.PurePosixPath(p).name != '.env.example' for p in paths)
    demo = 'media/emotion-classification-demo.mp4'
    assert demo in paths and (ROOT/demo).stat().st_size > 0, 'Missing demo recording'
    assert digest(git('show', ':'+demo)) == digest((ROOT/demo).read_bytes()), 'Unstaged demo changes'
    git('diff','--cached','--check')
    print(json.dumps({'status':'passed','hash_verified_checkout_files':checked,
        'frozen_notebook_runtime_files':notebooks,'staged_files':len(paths),
        'staged_bytes':sum((ROOT/p).stat().st_size for p in paths),
        'model_weights':'excluded from Git; supplied by pinned HF snapshot',
        'notebook_policy':'internal identities checked; notebooks remain frozen and may differ from current UI',
        'demo_present':True,'demo_path':demo},indent=2))


if __name__ == '__main__':
    main()
