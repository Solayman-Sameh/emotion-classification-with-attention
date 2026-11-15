import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

P = pathlib.Path(__file__).resolve().parents[1]
ROOT = P
sys.path.insert(0, str(P / 'application'))
import download_models as dm
from huggingface_hub.errors import LocalEntryNotFoundError

SCRATCH = ROOT / '.release-checks/fixtures'
SCRATCH.mkdir(parents=True, exist_ok=True)


def fixture(name, selected='distilbert'):
    root = SCRATCH / name
    root.mkdir(exist_ok=True)
    manifest = {}
    required = set(dm.REQUIRED_FILES)
    if selected != 'distilbert':
        required -= {'models/distilbert/config.json', 'models/distilbert/model.safetensors'}
        required.add('models/' + selected + '/best.pt')
    for name in sorted(required):
        data = (json.dumps({'selected': selected, 'attention_model': 'bilstm_attention'}).encode()
                if name == 'application.json' else b'{}' if name in {'config.json', 'vocab.json'}
                else ('fixture:' + name).encode())
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    (root/'files_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


class Downloads(unittest.TestCase):
    def test_recurrent_exports_resolve_locally_without_distilbert(self):
        for selected in ['lstm', 'gru', 'bilstm_attention']:
            with self.subTest(selected=selected):
                root = fixture('roles-' + selected, selected)
                with patch.object(dm, 'download_models', side_effect=AssertionError('unexpected network')):
                    self.assertEqual(dm.resolve_model_root(root), root)

    def test_selected_checkpoint_must_be_declared_in_local_manifest(self):
        root = fixture('missing-role', 'lstm')
        path = root/'files_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        del manifest['models/lstm/best.pt']
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'for selected model'):
            dm.resolve_model_root(root)

    def test_missing_recurrent_checkpoint_uses_partial_bundle_fallback(self):
        root = fixture('partial-lstm', 'lstm')
        (root/'models/lstm/best.pt').unlink()
        with patch.object(dm, 'download_models', return_value=pathlib.Path('cached')) as fallback:
            self.assertEqual(dm.resolve_model_root(root), pathlib.Path('cached'))
            fallback.assert_called_once()

    def test_recurrent_bundle_cannot_replace_default_remote_contract(self):
        root = fixture('remote-contract', 'lstm')
        with self.assertRaisesRegex(ValueError, 'Incomplete inference'):
            dm._manifest(root/'files_manifest.json')

    def test_model_role_metadata_is_hash_checked_before_use(self):
        root = fixture('corrupt-roles', 'lstm')
        (root/'application.json').write_text('{"selected":"distilbert"}', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Artifact identity mismatch: application.json'):
            dm.resolve_model_root(root)

    def test_complete_local_bundle_never_contacts_hub(self):
        root = fixture('local')
        with patch.object(dm, 'download_models', side_effect=AssertionError('unexpected network')):
            self.assertEqual(dm.resolve_model_root(root), root)

    def test_corrupt_local_file_fails_without_remote_substitution(self):
        root = fixture('corrupt')
        (root/'vocab.json').write_text('changed', encoding='utf-8')
        with patch.object(dm, 'download_models', side_effect=AssertionError('unexpected network')):
            with self.assertRaisesRegex(RuntimeError, 'Artifact identity mismatch'):
                dm.resolve_model_root(root)

    def test_partial_local_bundle_falls_back(self):
        root = fixture('partial')
        (root/'models/bilstm_attention/best.pt').unlink()
        with patch.object(dm, 'download_models', return_value=pathlib.Path('cached')) as fallback:
            self.assertEqual(dm.resolve_model_root(root, repo_id='owner/project'), pathlib.Path('cached'))
            self.assertEqual(fallback.call_args.args[0], 'owner/project')

    def test_download_cache_first_then_network_and_pin_one_revision(self):
        snapshot = fixture('a'*40)
        calls = []
        def fetch(**kw):
            calls.append(kw)
            if kw['local_files_only']:
                raise LocalEntryNotFoundError('not cached')
            return str(snapshot/kw['filename'])
        with patch('huggingface_hub.hf_hub_download', side_effect=fetch):
            self.assertEqual(dm.download_models('owner/project', cache_dir=SCRATCH), snapshot)
        self.assertEqual(calls[0]['revision'], 'main')
        self.assertTrue(all(c['revision']=='a'*40 for c in calls[2:]))
        self.assertTrue(all(calls[i]['local_files_only'] for i in range(0, len(calls), 2)))
        self.assertTrue(all(not calls[i]['local_files_only'] for i in range(1, len(calls), 2)))

    def test_cached_snapshot_works_offline(self):
        snapshot = fixture('b'*40)
        def fetch(**kw):
            self.assertTrue(kw['local_files_only'])
            return str(snapshot/kw['filename'])
        with patch('huggingface_hub.hf_hub_download', side_effect=fetch):
            self.assertEqual(dm.download_models('owner/project', local_files_only=True), snapshot)

    def test_offline_cache_miss_does_not_try_network(self):
        with patch('huggingface_hub.hf_hub_download', side_effect=LocalEntryNotFoundError('missing')) as fetch:
            with self.assertRaises(LocalEntryNotFoundError):
                dm.download_models('owner/project', local_files_only=True)
            self.assertEqual(fetch.call_count, 1)

    def test_corrupt_remote_file_is_rejected(self):
        snapshot = fixture('c'*40)
        (snapshot/'vocab.json').write_text('corrupt', encoding='utf-8')
        with patch('huggingface_hub.hf_hub_download', side_effect=lambda **kw: str(snapshot/kw['filename'])):
            with self.assertRaisesRegex(RuntimeError, 'Artifact identity mismatch'):
                dm.download_models('owner/project')

    def test_manifest_cannot_escape_snapshot(self):
        snapshot = fixture('unsafe')
        path = snapshot/'files_manifest.json'
        data = json.loads(path.read_text());data['../escape'] = '0'*64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Invalid inference manifest'):
            dm._manifest(path)

    def test_default_repo_is_pinned(self):
        snapshot = fixture('d'*40)
        with patch.dict(os.environ, {}, clear=True), patch('huggingface_hub.hf_hub_download', side_effect=lambda **kw: str(snapshot/kw['filename'])) as fetch:
            dm.download_models(cache_dir=SCRATCH)
            self.assertEqual(fetch.call_args_list[0].kwargs['repo_id'], dm.DEFAULT_HF_REPO_ID)
            self.assertEqual(fetch.call_args_list[0].kwargs['revision'], dm.DEFAULT_HF_REVISION)


class InferenceRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import types
        for name in ['common', 'models', 'inference']:
            module = types.ModuleType(name)
            module.__file__ = str(P/'application'/(name+'.py'))
            sys.modules[name] = module
            exec((P/'application'/(name+'.py')).read_text(encoding='utf-8'), module.__dict__)
        cls.inference = sys.modules['inference']

    def test_application_initializes_each_exported_model_role(self):
        for selected in ['distilbert', 'lstm', 'gru', 'bilstm_attention']:
            with self.subTest(selected=selected):
                root = fixture('application-roles-' + selected, selected)
                with patch.object(dm, 'download_models', side_effect=AssertionError('unexpected network')), \
                     patch.object(self.inference, 'load_model', return_value=(MagicMock(), None)):
                    application = self.inference.Application(root)
                self.assertEqual(set(application.loaded), {selected, 'bilstm_attention'})

    def test_local_transformer_never_resolves_remote_bundle(self):
        root = fixture('local-transformer')
        transformer = MagicMock()
        with patch.dict(sys.modules, {'transformers': transformer}), patch.object(self.inference, 'resolve_model_root', side_effect=AssertionError('unexpected download')):
            self.inference.load_model('distilbert', root/'models/distilbert', root/'tokenizer')
        self.assertEqual(transformer.AutoTokenizer.from_pretrained.call_args.args[0], root/'tokenizer')
        self.assertTrue(transformer.AutoModelForSequenceClassification.from_pretrained.call_args.kwargs['local_files_only'])
        self.assertFalse(transformer.AutoModelForSequenceClassification.from_pretrained.call_args.kwargs['trust_remote_code'])

    def test_missing_transformer_uses_resolved_hf_snapshot(self):
        root = fixture('cached-transformer')
        transformer = MagicMock()
        missing = SCRATCH/'missing-transformer-root'
        with patch.dict(sys.modules, {'transformers': transformer}), patch.object(self.inference, 'resolve_model_root', return_value=root) as resolver:
            self.inference.load_model('distilbert', missing/'models/distilbert', missing/'tokenizer', repo_id='owner/project')
        self.assertEqual(resolver.call_args.kwargs['repo_id'], 'owner/project')
        self.assertEqual(transformer.AutoModelForSequenceClassification.from_pretrained.call_args.args[0], root/'models/distilbert')

    def test_missing_attention_uses_same_fallback(self):
        root = fixture('cached-attention')
        with patch.object(self.inference, 'resolve_model_root', return_value=root), patch.object(self.inference, 'load_recurrent', return_value=('loaded', {})) as loader:
            model, tokenizer = self.inference.load_model('bilstm_attention', SCRATCH/'missing/models/bilstm_attention', SCRATCH/'missing/tokenizer')
        self.assertEqual(loader.call_args.args[0], root/'models/bilstm_attention/best.pt')
        self.assertEqual((model, tokenizer), ('loaded', None))


if __name__ == '__main__':
    unittest.main(verbosity=2)
