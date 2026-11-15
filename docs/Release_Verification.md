# Release verification

## Publication release — 15 November 2026

The application now uses the redesigned tone cards, colored word tiles, separate
word-focus prediction, explanatory popover and responsive layout. Example chips
fill the text box without classifying; Analyze Tone runs prediction. Loading,
model roles and prediction calculations retain their verified behavior.

The application requirements now pin the versions exercised in the Python
3.11.15 Conda main environment. Unused Plotly and SOCKS-proxy extras were removed
from the inference requirements. The training notebooks retain their independent
historical package specifications. No packages or model weights were downloaded
to prepare this release. A clean online installation is not claimed.

Current verification includes 22 loading/presentation tests, the real cached-model
HTTP smoke over seven inputs and blank rejection, desktop/mobile browser review,
application checksums, and the final staged-release verifier. The demo is
`media/emotion-classification-demo.mp4` (1920x1080, 30 fps, 35.7 seconds, 3.02 MiB).
Beginning, middle and final sample frames decoded successfully.

The notebook files remain frozen at the user's request. Release verification
checks each notebook's internal embedded hashes independently; it no longer
requires its historical app source to equal today's UI. This identity check does
not rerun notebook cells or resolve the workflow limitations documented in the
audit. No new training or test-set model evaluation is claimed.

The pinned public HF manifest and a one-byte range request for the model were
verified on 14 November. This establishes download-link access, not a complete
fresh-cache download. The full existing cache was used for real inference checks.

The root README is user-maintained and is excluded from this publication change.
The release is based on the existing GitHub main history; no force push is used.

The dated sections below are historical records, not current assertions of
notebook/application source equality or package versions.

## User UI redesign integration — 13 November 2026

Imported the user's revised app and attention renderer from Downloads. The UI now
uses two result cards, ranked probability bars, token-weight bars, example chips,
a Clear button and an expandable explanation. The root launcher applies the
custom theme and CSS; version-sensitive theme/HTML arguments are routed according
to the installed Gradio signatures. The attention card explicitly names BiLSTM +
Attention, and the fixed bar scale is disclosed as 25%.

All 22 loading/presentation regression tests passed. The real cached-model HTTP
smoke passed seven inputs and blank-input rejection with the two-card response.
Checks cover all six exact probabilities, ranked display, repeated token positions,
HTML escaping and long-input disclosure. Model loading and prediction functions
are unchanged; only attention_html changed in inference.py.

Application checksums and all 17 embedded notebook hashes are synchronized. Saved
notebook outputs and execution counts are preserved. Historical reports and the
original executed notebook remain unchanged. No dependencies or model weights
were installed or downloaded. Fresh dependency-install testing remains deferred.
The updated UI passed real HTTP API checks, but a fresh browser visual check was
not completed: the preview connection was refused. No new visual approval is claimed.


## Executed-notebook metadata correction — 13 November 2026

The original executed notebook's `cloud_evidence` now describes its saved full
Kaggle run instead of the obsolete smoke-only/no-CUDA claim. Its six stale
metadata hashes were recomputed from its own embedded runtime, which already
matched the notebook's internal hash dictionary. All 16 embedded hashes agree.

Only those two metadata fields changed. All 18 cells, including their source,
20 saved output objects, execution counts and cell metadata, are unchanged.
No cells, training or evaluation were rerun. The all-in-one notebook was not
modified by this metadata correction.

## UI review status

The user tested the earlier UI before submitting the redesign. User review of
the newly integrated redesign is pending. A prior assistant desktop browser spot-check during the project audit
successfully displayed the prediction chart and separately labeled attention
view; this does not constitute the user's review. Narrow/mobile rendering is
still unchecked. The browser failures recorded under the earlier release
checks below describe earlier attempts, before that successful spot-check.

## Model-role loader correction — 13 November 2026

Local exports now validate the selected classifier plus BiLSTM + Attention,
rather than requiring DistilBERT in every application. Application role metadata
is hash-checked before use. The default remote bundle requirements, revision pin,
path checks and artifact checksum checks remain enforced.

- All 18 loading regression tests passed, including recurrent-only exports,
  missing selected checkpoints, metadata corruption and application role routing.
- The notebook export function produced LSTM and BiLSTM + Attention applications
  using the existing trained checkpoints. Both loaded and predicted successfully
  without DistilBERT files. On this Windows test host, only the test copies' manifest
  separators were normalized to the POSIX paths produced natively on Kaggle.
- The real cached-model Gradio HTTP smoke passed all seven probes and blank-input
  rejection after the loader correction.
- All 17 embedded runtime hashes and present application artifact hashes passed.
  Only the embedded downloader source and corresponding hashes changed in the
  all-in-one notebook. Every cell output and execution count was preserved.
  The original executed notebook is byte-for-byte unchanged.

Full Kaggle Run All, retraining and held-out evaluation were not repeated. These
checks establish the corrected export/loading path, not a new training run.
The earlier staged-source checks below describe the 12 November source snapshot;
this correction is a subsequent working-tree change.

## Earlier release checks — 12 November 2026

Verified on 12 November 2026. Training was not repeated. The demo recording is
the remaining content deliverable. Environment-blocked checks below are still
unverified and must not be described as passed.

## Completed

| Check | Evidence |
|---|---|
| Loading regression tests | 12 unittest checks passed: local-first routing, cache/offline behavior, partial bundles, hash failures, pinned revision and model path routing |
| Git checkout integrity | All 20 locally present application artifacts match their manifests after Git's Windows checkout filters; .gitattributes enforces LF |
| Source export | Fresh source directory produced with git checkout-index from the staged index, without model weights or training archives |
| Real application smoke | App launched from that staged-source export, loaded the pinned cached models and served its actual Gradio HTTP API |
| Seven inputs | Positive text, negation, repeated tokens, emoji, a single token, 160-token text, and HTML-like text passed |
| Output checks | DistilBERT prediction; exact six-class bar order; probabilities sum to one; separately named attention prediction; normalized, escaped token weights |
| Edge cases | Blank input rejected; three repeated happy tokens retained separate positions; long input limited to 128 recurrent tokens with truncation disclosure |
| Notebook | 17 embedded file hashes synchronized with metadata; inference source copies match; seven saved result displays preserved |
| Packaging | Both application/experiment ZIP CRCs and internal manifests verified; unchanged model weights compared byte for byte |
| Git preparation | origin configured for the intended GitHub URL; branch main; no initial commit or push yet |
| Publication materials | README, proposal, demo guide and LinkedIn draft updated |

The negation probe produced a useful real disagreement: DistilBERT predicted
sadness for "I am not happy about this." while BiLSTM + Attention predicted joy.
The app retained that disagreement rather than relabeling one model's output.

## Reproduce available checks

From the project root, with the application dependencies installed:

```bash
python -m unittest discover -s tests -v
git add .
python tools/verify_release.py
python tools/smoke_app.py
```

The HTTP smoke requires the already cached HF model snapshot. It starts a local
server on port 7867, tests it, and stops it. It does not publish the app.

## Environment limits

- Outbound command-line network access is disabled in this execution environment.
  The configured proxy refuses connections. The HF online check failed before
  reaching the service; this is not evidence that the user's uploaded model is
  missing or invalid.
- A fresh online HF download was not completed. The existing snapshot
  `ee5352751c317e37e609630db3b8db0f5febd25e` is complete and verified.
- A clean dependency installation was not completed. An offline pip dry run
  could not obtain the required PyTorch wheel. Root and application requirements
  now share one dependency definition, but that is not a fresh-install test.
- Host smoke-test packages were Python 3.11.15, torch 2.5.1+cu121,
  transformers 5.13.0, gradio 6.30.0, huggingface-hub 1.16.0, numpy 2.4.6 and
  plotly 5.24.1. This host differs from the declared application dependency set.
  The original Kaggle/export checks used the pinned core package versions.
- Browser automation could not complete: Playwright was denied a Windows pipe;
  ChromeDriver could start but the sandboxed Chrome process crashed before a
  browser session was established. No desktop/mobile visual approval or browser
  screenshot is claimed. Confirm layout while recording the demo.
- GitHub remote existence/access could not be checked over the disabled network.
  The configured origin is the intended URL, not proof of a created cloud repo.

On a network-enabled machine, perform the clean-venv quickstart and then:

```bash
python tools/check_hf_online.py
python tools/check_hf_online.py --fresh-cache
```

The second command downloads approximately 274 MB into an explicitly empty
verification cache. Check desktop/narrow browser rendering and the README/video
links as part of the final recording/publication pass.

## Model and application release separation

The HF commit is an immutable model-asset snapshot. Current app code, README and
packaging can differ from its historical bundled Python files. The downloader
verifies that snapshot's own manifest and returns its model/tokenizer/data paths.
Inference executes this Git repository's code with remote code disabled; it does
not import Python from the downloaded snapshot. No new HF weight upload is
required for the local line-ending, documentation or source synchronization fixes.

The initial Git commit remains deferred so the finished video can join the
one-shot publication. See [Demo guide](Demo_Guide.md) and [LinkedIn draft](LinkedIn_Post.md).
