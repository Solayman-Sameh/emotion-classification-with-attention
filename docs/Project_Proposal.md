# Emotion Classification with Attention — Project Proposal

**Program:** Instant AI & Data Analysis Training
**Sprint:** 03 — RNN & NLP
**Project:** 03 — Emotion Classification with Attention
**Date:** 7 November 2026
**Status:** Full four-model training, data audit, held-out evaluation and inference packaging complete. The Hugging Face model snapshot is pinned and cache-verified. See `Release_Verification.md` for application checks and environment limits; demo recording is pending.

## 1. Project overview

This project develops an English text classifier that predicts one of six core emotions: **joy, sadness, anger, fear, surprise, and disgust**. Four architectures are trained and compared: LSTM, GRU, BiLSTM + Attention, and fine-tuned DistilBERT.

The system has two related outputs. The primary classifier predicts a dominant emotion and a six-class confidence distribution. The BiLSTM + Attention model additionally exposes normalized weights over its input tokens so that representative predictions can be inspected through heatmaps.

The final application uses Gradio, an allowed option in the assignment. It includes the selected classifier and preserves the attention model for the required visualization even if another architecture wins the comparison.

Cloud analysis and reproducible preparation lead to a small, self-contained Kaggle notebook. The notebook downloads all source assets inside Kaggle, repeats preparation, trains all four models, evaluates them, and creates inference and application archives. Only the notebook needs to be uploaded to Kaggle.

## 2. Assignment requirements

The authoritative source is **INSTANT AI & DATA ANALYSIS TRAINING.pdf**, pages **23–25**, under **Emotion Classification with Attention**. Its local path is `C:\Users\user\Documents\Antigravity\Instant_training\INSTANT AI & DATA ANALYSIS TRAINING.pdf`.

| Area | Required work |
|---|---|
| Dataset | Explore an emotion dataset such as GoEmotions or ISEAR |
| Label preparation | Map raw labels to the six target classes; required for GoEmotions |
| Text preparation | Clean, tokenize, and pad sequences to a fixed configured length |
| Embeddings | Initialize the recurrent embedding layer from pretrained GloVe or FastText vectors |
| Custom models | Train LSTM, GRU, and BiLSTM + Attention |
| Transformer | Fine-tune DistilBERT using the Hugging Face Trainer API |
| Evaluation | Accuracy, macro F1, per-class precision/recall, and confusion matrix |
| Attention | Plot per-token attention heatmaps for model predictions |
| Deployment | Streamlit or Gradio app showing dominant emotion and confidence bars across all six classes |

GoEmotions, GloVe-50, additive attention, Gradio, the filtering policy, and the execution settings below are project choices. Their exact source revisions, counts, dependency versions, and measured performance remain audit outputs rather than facts assumed in advance.

## 3. Dataset and six-class target

### 3.1 Primary source

The assignment links to [Google Research's GoEmotions dataset on Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions). The preferred configuration is **simplified**, which provides text, comment identifiers, lists of emotion labels, and existing train/validation/test splits.

GoEmotions has 27 fine-grained emotion categories plus neutral and can assign several labels to one text. It is not already the required six-class, single-label task. The project derives a clearly documented six-class subset rather than silently taking the first label or treating the original labels as the final taxonomy.

The dataset is already annotated. Manual labeling, synthetic target generation, and pseudo-labeling are outside the initial scope. ISEAR is an assignment-permitted alternative, but any change from the preferred source would be explicit and justified rather than automatic.

The source revision, configuration, downloaded filenames, schema, label-name metadata, file checksums, attribution, and applicable dataset terms are recorded during the cloud audit. Numeric source label IDs are read from metadata rather than hard-coded from memory.

### 3.2 Fine-grained label mapping

The initial mapping follows [Google Research's published Ekman mapping](https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json), pinned to a resolved repository revision during the audit.

| Target | Source emotion names |
|---|---|
| anger | anger, annoyance, disapproval |
| disgust | disgust |
| fear | fear, nervousness |
| joy | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring |
| sadness | sadness, disappointment, embarrassment, grief, remorse |
| surprise | surprise, realization, confusion, curiosity |

This is a published grouping convention, not a claim that all fine-grained emotions are psychologically identical. In particular, the interpretation of surprise includes the mapped cognitive categories and is explained in the report.

The task label inventory is exactly the six requested emotions. A different six-label corpus containing love instead of disgust would not satisfy the same target definition merely because its class count is six. Label names and order are persisted with every model and probability plot.

### 3.3 Single-label conversion policy

The proposed initial policy is deterministic and independent of model outcomes:

1. Validate the raw label list and resolve IDs to names from dataset metadata.
2. Exclude blank/unusable text and empty label lists, recording the reason.
3. Exclude examples containing neutral, including neutral mixed with emotion labels. This conservative choice prevents ambiguous neutral-plus-emotion annotations from entering the single-label subset.
4. Map the remaining fine-grained labels to target emotions and take the set of distinct mapped classes.
5. Retain an example when that set contains exactly one target class. Multiple source labels mapping to that same class are acceptable.
6. Exclude examples mapping to more than one target emotion. Do not choose one arbitrarily, duplicate the text with several single-label targets, or convert the task to a multi-label sigmoid classifier.
7. Treat unexpected label names or schema changes as errors requiring investigation rather than silently dropping them.

For example, joy plus amusement maps unambiguously to joy; anger plus sadness is excluded. These examples illustrate policy only and do not assert observed counts.

The audit records retention and exclusion totals by reason, source label combination, resulting class, and original split. It also discusses selection bias: performance on the retained unambiguous subset does not establish performance on all mixed-emotion comments.

## 4. Split integrity, duplicates, and class balance

Existing source split membership is the starting point. The project does not merge the source splits and randomly redistribute examples simply for convenience. All four architectures use exactly the same final retained record IDs and labels.

Duplicate IDs, normalized-text duplicates, conflicting mapped labels, and overlaps between source splits are inspected before training. The proposed leakage-controlled policy removes conflicting full-text groups and resolves consistent cross-split duplicates with fixed priority **test, then validation, then training**, removing lower-priority copies without moving records between splits. Consistent duplicates within a split retain a stable representative. Every exclusion is logged.

Identity normalization is versioned separately from model preprocessing and preserves meaningful distinctions where possible. The final report describes the result as a filtered, leakage-controlled GoEmotions-derived task, not an unchanged published benchmark. Collisions introduced by aggressive normalization, unknown-token encoding, or truncation are audited; any additional exclusion/length adjustment uses a predeclared rule without looking at model test scores. Zero-overlap claims require actual verification.

Training-only data determines word vocabulary, any learned preprocessing settings, embedding matrix rows for the task vocabulary, and class weights. Validation guides tuning, early stopping, and model selection. The real test partition is used only for the final frozen reporting phase; descriptive quality checks are separate from predictive test evaluation.

The six classes are unlikely to be equally represented after mapping. Their actual supports must be measured, especially disgust and fear. Training-derived class-weighted cross-entropy is the initial imbalance treatment when warranted by those counts. Weights use a documented formula, are applied consistently across the four models where feasible, and are checked for numerical stability. Oversampling is not automatically combined with weighting. Validation/test prevalence is not artificially balanced, and rare classes are not removed to improve headline metrics.

## 5. Dataset audit and EDA

The cloud analysis and notebook EDA will report:

- Source revision/configuration, schema, raw split sizes, and original label metadata.
- Multi-label frequencies, neutral frequency, mapping validity, and label combinations.
- Exclusion flow and final records per split and target emotion.
- Class distributions before/after mapping and any imbalance treatment.
- Duplicate/conflict/overlap findings and the final split policy.
- Text lengths in characters, recurrent tokens, and transformer subwords.
- Coverage of pretrained embeddings, unknown-token rates, and truncation rates.
- Representative retained examples for each class and examples of excluded ambiguous labels.
- A reproducibility record containing source IDs, mappings, vocabulary, split assignments, code/dependency versions, and hashes.

Exact retained counts and expected fingerprints are created from actual downloaded data, not invented in the proposal. Figures and tables in the final notebook are reproducible outputs rather than static screenshots standing in for executed analysis.

## 6. Text and embedding preparation

Emotion recognition benefits from negation, contractions, punctuation, and expressive tokens. The initial cleaning is minimal: validate text, normalize Unicode/whitespace through a documented policy, and handle URLs or user markers consistently if the audit justifies it. Stop-word removal and stemming are not copied automatically from complaint classification because phrases such as “not happy” would lose meaning.

The original text is retained. The recurrent tokenizer has a deterministic policy for words, contractions, punctuation, emoji, and redaction markers, with examples and OOV statistics. Lowercase lookup can match uncased GloVe vectors, but the actual preprocessing and any lost signals are documented. DistilBERT uses its own tokenizer on minimally modified source text, rather than receiving aggressively cleaned word lists.

The preferred pretrained embedding resource is **GloVe-50**, with [Gensim-data's glove-wiki-gigaword-50](https://github.com/piskvorky/gensim-data) as a compact acquisition candidate. A verified direct vector-file streaming route can avoid installing Gensim only for conversion. The downloaded resource's provenance, format, dimension, checksum, and resolved source are recorded.

The task embedding matrix retains only rows needed for the training vocabulary. Unmatched words receive seeded unknown/random initialization with coverage reported; the entire word matrix is not silently replaced with random initialization if the pretrained resource fails. Padding has a zero vector and does not affect sequence states or attention. The initial embedding layer is fine-tunable under the same policy for all three recurrent models.

Recurrent sequences use a fixed configured maximum length and padding as required. Valid lengths/masks exclude padding from state extraction and pooling. A provisional length of **128 tokens** is a candidate, with final word/subword caps selected from the training audit and memory measurements. Word tokens and transformer subwords are not treated as interchangeable units. Truncation fractions and the head/tail policy are reported, and app heatmaps show only the tokens actually used.

## 7. Four required models

| Model | Initial design | Purpose |
|---|---|---|
| LSTM | Pretrained embedding → unidirectional LSTM → valid final state → dropout → six-class linear head | Sequential baseline |
| GRU | Same embedding policy → unidirectional GRU → valid final state → dropout → six-class linear head | Gated recurrent comparison |
| BiLSTM + Attention | Pretrained embedding → bidirectional LSTM → masked learned additive attention → weighted context → dropout → six-class linear head | Attention-based sequence aggregation and token heatmaps |
| DistilBERT | Pretrained uncased DistilBERT encoder with a six-class classification head | Fine-tuned transformer benchmark |

The transformer candidate is [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased), pinned to a resolved revision. Fine-tuning uses **AutoTokenizer, AutoModelForSequenceClassification, and Hugging Face Trainer**. Encoder parameters receive gradients. Pretrained inference or training only a head on a frozen encoder does not replace the assignment's fine-tuning experiment.

The custom classifiers may use standard PyTorch LSTM and GRU layers. Neither character-level NER features nor CRF layers are required here; they are specific to the separate NER design.

### 7.1 Attention mechanism

Additive attention assigns each valid recurrent hidden state a learned score, applies a masked softmax over valid positions, and pools the hidden states using those normalized weights. It is trained jointly with the classifier through the emotion-classification loss.

Verification checks that padded positions receive zero weight, valid-token weights sum to one within tolerance, outputs and gradients remain finite, and padding does not change predictions in evaluation mode. Empty/all-padding inputs are rejected before softmax to avoid invalid normalization.

The heatmap visualizes weights from this exact trained module and its input-token sequence. It does not substitute gradient scores, random colors, or transformer attention for the required BiLSTM attention weights. Attention is an inspection aid rather than proof of causal importance or a complete explanation of a prediction.

Because the required LSTM baseline and BiLSTM + Attention differ in directionality as well as pooling, their score difference cannot be attributed exclusively to attention. A matched BiLSTM without attention is an optional additional control, not a fifth required experiment within the initial deadline.

### 7.2 Initial training configuration

| Setting | Initial candidate, subject to audit/pilot verification |
|---|---|
| Seed | 42 |
| Recurrent embedding | Pretrained 50-dimensional GloVe, initially fine-tunable |
| Hidden size | 128 per recurrent direction |
| Recurrent depth | One layer |
| Attention projection | Approximately 128 units |
| Dropout | 0.3 outside the recurrent layer |
| Recurrent optimizer | Adam; learning rate 0.001 |
| Transformer optimizer | Trainer-managed AdamW; learning rate 0.00002 |
| Recurrent epoch ceiling | 10; early stopping patience 2 |
| Transformer epoch ceiling | 3; early stopping patience 2 |
| Initial training batches | Recurrent 64; transformer 16, with accumulation selected from memory budget |
| Stability | Clipping, padding masks, supported mixed precision, finite-loss/gradient checks |
| Selection metric | Validation macro F1 |

These are starting points, not measured best settings. One full configuration per required architecture is the initial scope. Large searches, hidden category removal, synthetic data generation, and repeated seed sweeps are deferred.

## 8. Evaluation and model selection

Every model reports accuracy, macro F1, macro/weighted precision and recall, per-class precision/recall/F1/support, and raw-count and row-normalized confusion matrices. Class support is always shown because a rare class's F1 can be unstable.

Learning curves, best epoch, training time, GPU assignment, model/parameter size, peak memory, and measured inference latency are recorded. The report distinguishes inference-only timing from preprocessing-plus-inference timing. Runs performed concurrently are labeled as concurrent rather than isolated hardware benchmarks.

The highest validation macro F1 selects the deployment winner. The proposed tie rule considers models within **0.001 absolute macro F1** equivalent, preferring fewer trainable parameters and then measured validation inference latency. This policy is frozen before opening model test outcomes.

After the experiment configuration and selection are frozen, all four best checkpoints receive one final test-evaluation phase. Test results do not trigger winner replacement or retuning within that same held-out claim.

Error analysis covers confusing classes, minority-class errors, negation, sarcasm or ambiguous examples, OOV tokens, and truncated inputs. Generalization limits of the filtered single-label corpus and the chosen label mapping are explicit. No fixed accuracy/F1 target is promised before training.

## 9. Attention visualizations and application

### 9.1 Two named prediction roles

The **selected classifier** supplies the main dominant emotion and six-class confidence bars. The **BiLSTM + Attention model** supplies its own token weights, its own predicted emotion, and a separately labeled inspection view.

If the attention model wins, the main prediction and heatmap refer to the same model. If DistilBERT, LSTM, or GRU wins, the application still retains the attention checkpoint and clearly labels that the heatmap belongs to a different model. It does not present BiLSTM weights as an explanation of another model's decision. Predictions can disagree and that disagreement remains visible.

### 9.2 Display behavior

The Gradio app includes a text box, classification action, predicted emotion, six-class probability chart, a token-level attention view, examples, and clear model labels. Attention processing can be on demand so it does not unnecessarily delay every main-model prediction.

The heatmap preserves token order, repeats separate occurrences of the same word rather than collapsing them, and labels the processed-token representation. Padding is hidden. Any truncation or omitted text is disclosed. The displayed numerical attention weights are the actual normalized weights, not per-example min/max rescaling represented as probabilities. Exportable heatmap examples accompany the report.

Blank input, text becoming unusable after preprocessing, emoji/punctuation-heavy input, repeated tokens, single-token input, and long input are handled explicitly. HTML-based highlighting escapes user text before rendering.

Confidence bars display model probabilities, with calibration limitations stated. Class-weighted training does not automatically produce calibrated probabilities. The six-class system has no neutral output; neutral, mixed-emotion, or out-of-domain text can still be assigned a target emotion. Threshold-based rejection is optional and requires validation rather than an invented cutoff. The app describes expressed textual emotion, not a person's mental state or a clinical assessment.

### 9.3 Inference packaging

The app bundle contains the selected classifier and the attention model when they differ, plus exact tokenizers/vocabularies, trained embedding weights, architecture/configuration, labels and their order, preprocessing, maximum lengths, and heatmap logic. Checkpoint hashes and a manifest identify each model's role.

A clean CPU reload must reproduce evaluation predictions within a documented tolerance. The attention checkpoint also reproduces its saved weights/heatmaps. The app requires neither the full dataset nor a fresh full GloVe download at startup. Hosting/publication is a separate action from creating the trained application archive.

## 10. All-in-one Kaggle notebook and cloud workflow

Cloud analysis first downloads and inspects the real sources, validates the mapping and preparation, measures embedding coverage, and tests functional model paths. The final notebook includes those EDA/preparation steps as executable reproducible code rather than requiring a locally uploaded prepared dataset.

The normal Kaggle workflow is:

1. Import the small notebook.
2. Enable Internet.
3. Select GPU T4 ×2, or a single available GPU.
4. Save Version and Run All.

All necessary runtime modules are embedded. Dataset/embedding/model sources are downloaded directly within Kaggle. Large assets are not embedded in notebook JSON and are not routed through the user's computer. Live source access, consent/authentication requirements if any, dependency versions, and hashes are verified rather than assumed. Source changes and incompatible caches stop with actionable diagnostics.

The notebook includes configuration, source download, mapping/filtering, EDA, splitting and deduplication evidence, preprocessing, embeddings/tokenizers, bounded verification, four-model training, validation selection, final test evaluation, attention figures, clean inference exports, and output archives.

### 10.1 Parallelism within the project

Independent worker processes run at most one architecture on each available GPU. DistilBERT starts alongside a recurrent architecture; remaining custom models are queued for the next free device. GPU visibility is set before importing PyTorch and initializing Trainer, so a transformer worker cannot accidentally use both devices while a second worker occupies one.

Batch sizes and learning rates keep their per-model meaning. Single-GPU fallback, low CPU-thread counts, useful epoch/step logs, assignment records, worker-failure cancellation, and compatible completed-model reuse are included. Heartbeats indicate live workers; training steps/epochs and GPU activity provide stronger evidence of progress.

### 10.2 Parallelism across all three sprint projects

Each project can have a separate GPT Work cloud conversation, its own source audit, proposal, implementation, and notebook. Preparation, implementation, documentation, and results review can proceed concurrently while another project trains.

Project directories, notebook-generated working paths, cache identities, logs, model names, and output archive names are distinct. Project 3 uses emotion-specific names such as `emotion_classification_run/`; it does not write complaint or NER outputs or overwrite their files. Shared ideas and verified download resources can be reused, but no untested cross-project writable cache or shared filesystem is assumed between cloud conversations.

Actual simultaneous GPU training follows the account's available allocation and quota. Three independent T4×2 sessions would require six GPUs; access to a two-GPU option alone does not establish that availability. Each notebook is independently runnable on one or two assigned GPUs. A GPU allocated to another notebook is not automatically a device that this notebook can borrow. A combined cross-project GPU scheduler would be a separate integration task, not a claimed feature of these independent notebooks.

Project 1's active training can continue while Projects 2 and 3 are prepared in the cloud. Their full GPU runs start when suitable resources are available, or run concurrently only when the platform actually allocates those resources. No fixed total completion time is promised without measured throughput and available quota.

### 10.3 Dependency, cache, and restart policy

The dependency set is validated together, including PyTorch, Transformers, Trainer/Accelerate, source loaders, embeddings, plotting, and Gradio inference. An isolated project runtime for preparation and training workers is preferred where preloaded notebook libraries conflict, while retaining compatible CUDA PyTorch access.

The Project 1 startup failure pattern is explicitly avoided: a notebook cannot demand a restart before installing needed packages, and a factory reset cannot be required to apply an installation that it would erase. A necessary kernel-only restart is documented and tested as a narrow exception. A recurring process-exit cell is not part of the normal Run All path. Unrelated package warnings are distinguished from actual failed project imports or installation errors.

Preparation and completed models are reused only when source, mapping, filtering, code, preprocessing, embedding/tokenizer, dependency, and configuration identities match. A new Kaggle session does not automatically restore working files; restoration from saved outputs is explicit. Optimizer-state resumption is claimed only if optimizer, scheduler, scaler, RNG/progress state, and interruption recovery are actually implemented and tested. Otherwise an interrupted model restarts, while completed compatible models can be reused.

## 11. Verification and truthful evidence

| Area | Required evidence |
|---|---|
| Sources | Actual dataset, mapping, embedding, and model/tokenizer access with revisions and checksums |
| Mapping | Same-group multi-label retention; mixed-group exclusion; neutral handling; unknown-label error; exact six-class order |
| Data | Record ledger, retained supports, duplicates/overlaps, fixed shared splits, training-only vocabulary and weights |
| Embeddings | Real pretrained-vector loading, shape/coverage, padding row, seeded unmatched rows |
| Sequence models | Masking, padding invariance, finite loss, actual gradients, reproducible reload |
| Attention | Correct token count/order, zero padding mass, normalized valid weights, repeated/single-token handling, finite gradients |
| Transformer | Trainer path, actual encoder fine-tuning, correct labels, tokenizer and mask behavior |
| Scheduling | Cross-device overlap, one job per GPU, visibility isolation, single-GPU fallback, failure cancellation |
| Integration | Tiny real train/validation-only runs for all four models; selected plus attention-model export; archives |
| App | Correct six bars/label order, explicit model roles, escaped token display, long/blank input, clean CPU inference |
| Notebook | Schema/syntax, embedded-source identity, correct cell order, fresh-process/native-kernel execution as available |

Full mode is the default, preceded by bounded checks. Smoke mode uses training/validation samples only and never evaluates the real held-out test set or presents its tiny metrics as quality results. Any preparation-only mode is named separately.

Cloud CPU execution and simulated scheduling tests do not establish real two-GPU CUDA execution. Reports distinguish code inspection, static checks, dummy-process tests, CPU neural runs, native notebook runs, actual CUDA runs, and full training. Unsupported environment checks are listed with the smallest necessary workaround. No expected metrics or source fingerprints are fabricated.

## 12. Deliverables, milestones, and scope

| Location or artifact | Purpose |
|---|---|
| `docs/Project_Proposal.md` | This descriptive project specification |
| `docs/Cloud_Work_Prompt.md` | Separate prompt requesting cloud implementation |
| `notebooks/emotion_classification_all_in_one.ipynb` | Self-contained Kaggle download-to-export workflow |
| `src/` and `configs/`, if retained | Reusable implementation and recorded settings; runtime code also embedded in notebook |
| Dataset/embedding/mapping audit | Source evidence, exclusion flow, figures, label distribution, and reproducibility record |
| Lightweight reports archive | Four-model comparison, diagnostics, verification, and attention figures without weights |
| Complete experiment archive | Four best checkpoints, sources/configuration references, logs, and results |
| Gradio application archive | Selected classifier, attention checkpoint when needed, and standalone app |
| Reproduction instructions | Exact Kaggle settings, rerun/restoration behavior, output retrieval, and app deployment steps |

Completion milestones are a real source audit; verified six-class preparation; functional four-model pipeline; one reproducible Kaggle notebook; validation-frozen model comparison; clean main/attention-model inference; and verified output packaging.

The proposed choices are sufficient to begin without more user preferences. Exact source revisions, retained counts, lengths, coverage, batch sizes, dependency versions, and timing are finalized from actual audit and pilot evidence. Extensive tuning, multilingual support, clinical interpretation, synthetic annotation, a cross-project GPU scheduler, and a causal-explanation study are outside the initial scope.

## 13. References

- **INSTANT AI & DATA ANALYSIS TRAINING.pdf**, pages 23–25: assignment requirements.
- [Google Research GoEmotions on Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions).
- [Google Research Ekman mapping](https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json).
- [GoEmotions original repository](https://github.com/google-research/google-research/tree/master/goemotions).
- [DistilBERT base uncased model card](https://huggingface.co/distilbert/distilbert-base-uncased).
- [Hugging Face text-classification/Trainer guide](https://huggingface.co/docs/transformers/tasks/sequence_classification).
- [Gensim-data pretrained-vector resources](https://github.com/piskvorky/gensim-data).

Public source descriptions were checked during proposal preparation. The completed Kaggle run, reproducible audit, trained checkpoints and held-out results are preserved in the executed notebook and reports. Release verification and environment limits are recorded in `Release_Verification.md`.
