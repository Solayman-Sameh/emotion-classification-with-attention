# Emotion classification application

Selected classifier: distilbert. Attention model: bilstm_attention. Mode: full.

The inference runtime is verified on Python 3.11.15. Install with `pip install -r requirements.txt`, then
`python app.py`. A complete local bundle runs offline. A source-only checkout
retrieves missing inference assets from the default HF snapshot. No dataset or
full GloVe download is performed. Hosting is a separate action.

Application dependencies match the tested local inference environment. The frozen
Kaggle notebook retains its own historical training dependencies. A clean online
installation has not been repeated, to avoid downloading large packages again.

Default repo: `Solayman-Sameh/emotion-classification-with-attention`.
Default revision: `ee5352751c317e37e609630db3b8db0f5febd25e`.
Default cache: `~/.cache/emotion_models`.

Override with `EMOTION_HF_REPO_ID`, `EMOTION_HF_REVISION`, or
`EMOTION_MODEL_CACHE`. A different repo defaults to `main` unless a revision is
supplied. Private repos use `hf auth login` or the host's `HF_TOKEN`.

Preload: `python download_models.py`. Check cache: `python download_models.py --offline`.

Local files are checked first, then the HF cache. Cache misses trigger downloads;
all files use one resolved commit and are hash-verified. A mismatch stops loading.
Weights are not copied into another cache. Transformers reads resolved local
assets with remote code disabled; app code comes from this checkout.

Bars are uncalibrated selected-model probabilities. The heatmap shows the separate
attention model's own prediction and normalized token weights. Neutral/mixed text
is forced into six labels. Notices and provenance are in `licenses/` and
`source_lock.json`.
