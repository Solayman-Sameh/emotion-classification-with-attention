"""Resolve a complete local inference bundle, then fall back to the HF model repo.

Set EMOTION_HF_REPO_ID=owner/repository or pass --repo-id. Authentication is
managed by `hf auth login` or HF_TOKEN; tokens are never stored in this project.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re

REQUIRED_FILES = (
    'models/distilbert/config.json', 'models/distilbert/model.safetensors',
    'models/bilstm_attention/best.pt', 'tokenizer/config.json',
    'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json',
    'tokenizer/vocab.txt', 'vocab.json', 'config.json', 'application.json',
    'common.py', 'models.py',
)


DEFAULT_HF_REPO_ID = 'Solayman-Sameh/emotion-classification-with-attention'
DEFAULT_HF_REVISION = 'ee5352751c317e37e609630db3b8db0f5febd25e'


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path, required_files=REQUIRED_FILES):
    data = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not set(required_files) <= set(data):
        raise ValueError('Incomplete inference files_manifest.json')
    for name, digest in data.items():
        relative = pathlib.PurePosixPath(name)
        if (not name or relative.is_absolute() or '..' in relative.parts
                or '\\' in name or ':' in name or not re.fullmatch(r'[a-f0-9]{64}', str(digest))):
            raise ValueError('Invalid inference manifest entry: ' + name)
    return data


def _local_manifest(path):
    # Local exports contain the selected classifier plus the attention model.
    # The downloadable default bundle still uses REQUIRED_FILES unchanged.
    shared = set(REQUIRED_FILES) - {
        'models/distilbert/config.json', 'models/distilbert/model.safetensors'}
    manifest = _manifest(path, shared)
    metadata = pathlib.Path(path).parent / 'application.json'
    if not metadata.is_file():
        return manifest  # Partial bundle: resolve_model_root handles fallback.
    if _hash(metadata) != manifest['application.json']:
        raise RuntimeError('Artifact identity mismatch: application.json')
    roles = json.loads(metadata.read_text(encoding='utf-8'))
    selected = roles.get('selected')
    if (selected not in {'distilbert', 'lstm', 'gru', 'bilstm_attention'}
            or roles.get('attention_model') != 'bilstm_attention'):
        raise ValueError('Invalid inference application model roles')
    selected_files = ({'models/distilbert/config.json', 'models/distilbert/model.safetensors'}
                      if selected == 'distilbert' else {'models/' + selected + '/best.pt'})
    if not selected_files <= set(manifest):
        raise ValueError('Incomplete inference files_manifest.json for selected model')
    return manifest


def _verify(root, manifest):
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or _hash(path) != expected:
            raise RuntimeError('Artifact identity mismatch: ' + name)


def download_models(repo_id=None, revision=None, cache_dir=None, local_files_only=False):
    """Download one pinned snapshot into the HF cache and verify its SHA-256s.

    Cached files are tried without network access first. A cache miss triggers
    a Hub download unless local_files_only=True. Returned paths stay in the
    managed cache; model weights are not copied into a second directory.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo_id = repo_id or os.environ.get('EMOTION_HF_REPO_ID') or DEFAULT_HF_REPO_ID
    revision = revision or os.environ.get('EMOTION_HF_REVISION') or (
        DEFAULT_HF_REVISION if repo_id == DEFAULT_HF_REPO_ID else 'main')
    cache_dir = pathlib.Path(cache_dir or os.environ.get(
        'EMOTION_MODEL_CACHE', str(pathlib.Path.home() / '.cache' / 'emotion_models')))

    def fetch(name, pinned_revision):
        kwargs = dict(repo_id=repo_id, filename=name, repo_type='model',
                      revision=pinned_revision, cache_dir=str(cache_dir))
        try:
            return pathlib.Path(hf_hub_download(**kwargs, local_files_only=True))
        except LocalEntryNotFoundError:
            if local_files_only:
                raise
            return pathlib.Path(hf_hub_download(**kwargs, local_files_only=False))

    manifest_path = fetch('files_manifest.json', revision)
    # HF returns cache/snapshots/<commit>/files_manifest.json. Keep that snapshot
    # path (do not resolve symlinks into blobs) so every file uses one revision.
    root = manifest_path.parent
    commit = root.name
    if not re.fullmatch(r'[a-f0-9]{40,64}', commit):
        raise RuntimeError('Expected a commit-addressed Hugging Face snapshot path')
    manifest = _manifest(manifest_path)
    for name in manifest:
        downloaded = fetch(name, commit)
        if downloaded != root / name:
            raise RuntimeError('Downloaded file is outside the selected model snapshot: ' + name)
    _verify(root, manifest)
    return root


def resolve_model_root(root, repo_id=None, revision=None, cache_dir=None, local_files_only=False):
    """Use a complete verified application directory before consulting the Hub.

    A partial directory falls back to a complete remote snapshot. Corrupt local
    files fail loudly rather than silently switching model versions.
    """
    root = pathlib.Path(root)
    manifest_path = root / 'files_manifest.json'
    if manifest_path.is_file():
        manifest = _local_manifest(manifest_path)
        if all((root / name).is_file() for name in manifest):
            _verify(root, manifest)
            return root
        for name, expected in manifest.items():
            path = root / name
            if path.is_file() and _hash(path) != expected:
                raise RuntimeError('Artifact identity mismatch: ' + name)
    return download_models(repo_id, revision, cache_dir, local_files_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id')
    parser.add_argument('--revision')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true', help='Use an already cached snapshot only')
    args = parser.parse_args()
    print(download_models(args.repo_id, args.revision, args.cache_dir, args.offline))


if __name__ == '__main__':
    main()
