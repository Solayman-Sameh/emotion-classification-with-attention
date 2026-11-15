"""Create the named HF model repo and upload the verified inference application.

Usage: python upload_to_hf.py --repo-id HF_OWNER/EXACT_GITHUB_REPO_NAME
Uses existing `hf auth login` / HF_TOKEN authentication. New repos are private
unless --public is supplied. Existing repository visibility is unchanged.
"""
import argparse
import hashlib
import io
import json
import pathlib
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True, help='HF owner/name; name must match the GitHub repository')
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--archive', type=pathlib.Path, default=pathlib.Path(__file__).parent / 'codex-space/emotion_trained_application.zip')
    args = parser.parse_args()
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(args.repo_id)
    if '/' not in args.repo_id:
        parser.error('--repo-id must include the Hugging Face owner and repository name')
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read('files_manifest.json'))
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError('Artifact identity mismatch: ' + name)
        names = sorted(manifest) + ['files_manifest.json']
        print(json.dumps({'repo_id': args.repo_id, 'files': names,
                          'bytes': sum(archive.getinfo(name).file_size for name in names),
                          'dry_run': args.dry_run}, indent=2))
        if args.dry_run:
            return
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        api.whoami()  # Fail before creating anything if login is unavailable.
        api.create_repo(repo_id=args.repo_id, repo_type='model', private=not args.public, exist_ok=True)
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=io.BytesIO(archive.read(name))) for name in names]
        commit = api.create_commit(repo_id=args.repo_id, repo_type='model', operations=operations,
                                   commit_message='Upload verified six-emotion inference bundle and local-first loader')
        info = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
        remote = {item.rfilename: item for item in info.siblings}
        for name in names:
            if name not in remote or remote[name].size != archive.getinfo(name).file_size:
                raise RuntimeError('Remote upload size verification failed: ' + name)
        print(json.dumps({'url': 'https://huggingface.co/' + args.repo_id, 'revision': commit.oid,
                          'verified_files': len(names), 'next_step': 'Set EMOTION_HF_REPO_ID and EMOTION_HF_REVISION in the app environment.'}, indent=2))


if __name__ == '__main__':
    main()
