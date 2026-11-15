# Cloud implementation handoff — Emotion Classification with Attention

Use the attached `Project_Proposal.md` as the specification for Project 3 of Sprint 03: Emotion Classification with Attention. Complete the cloud data analysis, implementation, and available verification, then deliver one small self-contained Kaggle notebook. The proposal describes the project; this prompt requests execution.

## Environment and parallel project work

First verify where your execution runs. Large dataset, embedding, and pretrained-weight downloads and model execution belong in a confirmed cloud sandbox or Kaggle. My local storage and bandwidth are limited. If your tools only execute on my computer, state that before downloading large assets or running models locally.

This is independent work alongside Project 1 (complaint classification) and Project 2 (NER), potentially in separate GPT Work cloud conversations. Use emotion-specific directories, source identifiers, configuration, logs and archive names. Do not overwrite their assets or assume a filesystem shared across conversations. Reuse sound infrastructure ideas without copying their task-specific preprocessing or labels.

Parallelize preparation and implementation across projects. The notebook itself should use up to two assigned GPUs through independent-model workers. Do not assume that access to T4×2 permits three simultaneous two-GPU sessions; actual concurrent training follows available allocation and quota. A combined scheduler spanning all three projects is not required.

The final normal workflow is: import the notebook, enable Internet, select GPU T4×2, and Save & Run All. Kaggle downloads all source assets and performs preparation itself. No prepared dataset ZIP, project folder, tokenizer, embedding file, or external scripts should need uploading to Kaggle. Embed all runtime code and small real reference metadata, not large data or weights.

## Real source audit and six-class preparation

Use `google-research-datasets/go_emotions`, configuration `simplified`, from https://huggingface.co/datasets/google-research-datasets/go_emotions. Inspect actual files, metadata, official splits, source revision, terms, and hashes. No silent dataset substitution or synthetic/manual relabeling.

Use the published mapping at https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json, verify it and pin its resolved revision. The target classes must be exactly joy, sadness, anger, fear, surprise, and disgust. Do not substitute a different six-label dataset whose taxonomy contains love instead of disgust.

Implement the proposal's explicit single-label policy: resolve raw label IDs from metadata; exclude empty labels and unusable text; exclude any neutral-bearing example; map remaining labels; retain multiple fine-grained labels only when they all map to one target class; exclude cross-emotion combinations. Unknown labels/schema changes require investigation. Never select the first source label or duplicate a mixed-emotion sample under several single-label targets.

Produce EDA covering raw/mapped distributions, multi-label combinations, neutral and ambiguity exclusions, supports by split, duplicates/conflicts/overlaps, token/subword lengths, pretrained coverage, OOV and truncation. Preserve source split assignments for retained rows and apply the fixed duplicate/exclusion policy described in the proposal. Record all changes and distinguish the derived task from an unchanged published benchmark. All four models use the same final record IDs and targets.

Fit task vocabulary and class weights from training data only. Preserve negation, contractions and emotion-bearing punctuation; do not blindly apply complaint-style stop-word removal. Keep original text, documented recurrent tokenization, and a minimally modified transformer input. Acquire real pretrained GloVe vectors, initially the compact 50-dimensional route, and report coverage. Do not silently replace failed pretrained acquisition with entirely random embeddings.

Generate expected counts and reference fingerprints from actual observed cloud preparation, then verify the notebook's regenerated preparation against them. Do not invent source hashes or retained class counts. Cache reuse must validate source, mapping, filtering, code, preprocessing, embeddings/tokenizer, dependencies and configuration identities.

## Four required architectures and training

Implement and train LSTM, GRU, BiLSTM + Attention, and genuinely fine-tuned `distilbert/distilbert-base-uncased`. The transformer must use Hugging Face AutoTokenizer, AutoModelForSequenceClassification and Trainer, with encoder gradients enabled.

Follow the proposal's initial training budget and document audit/pilot adjustments. Use pretrained recurrent embeddings, padding-aware state handling, stable masked additive attention, clipping, early stopping, best-checkpoint retention, and supported mixed precision. Address measured class imbalance consistently with training-only information. Do not drop rare target classes or reduce data silently to get faster results.

Verify actual attention weights: zero mass on padding, unit mass across real tokens, correct order and repeated-token identity, finite loss/gradients, and safe empty/all-padding handling. Keep the attention module trained jointly through classification loss. Do not substitute transformer attention or gradient saliency for the required BiLSTM attention visualization.

Use isolated subprocess workers, one model per assigned GPU. DistilBERT starts alongside a recurrent model, then queued models occupy freed devices. Set GPU visibility before PyTorch imports and Trainer initialization so the transformer worker cannot accidentally use both GPUs. Include one-GPU fallback, limited CPU threads, progress/logs, failure cancellation, and compatible completed-model reuse. Keep effective batch sizes/learning rates per model; two GPUs do not automatically imply scaling them.

## Evaluation and truthful attention display

Select the winner using validation macro F1 with the proposal's predeclared tie policy. Freeze selection before final held-out evaluation of all four best checkpoints. Report accuracy, macro F1, per-class precision/recall/F1/support, weighted/macro summaries, raw/normalized confusion matrices, curves, timings, memory and checkpoint information. Analyze minority-class errors, confusion, negation, OOV/truncation and label-filtering limitations. Do not tune on final test results.

Retain and export the trained BiLSTM + Attention even if it is not the winning classifier. The main app prediction and six confidence bars come from the selected model. A separately labeled attention view comes from BiLSTM + Attention and shows that model's own prediction. If they disagree, show that honestly; do not present one model's weights as an explanation of another model's decision.

Build a standalone Gradio app with the six-class chart, examples and token-level heatmaps. Preserve token order/occurrences, hide padding, show truncation, escape user text in HTML, and handle blank, short, emoji-heavy and long input. Probabilities and attention weights are not calibrated certainty or proof of causal word importance. Neutral/mixed input can be forced into one of the six classes; explain this limitation without inventing an unvalidated rejection threshold.

Export the selected model plus the attention model when different, along with exact tokenizers/vocabularies, trained embeddings, architecture, labels/order, preprocessing and heatmap code. The app must not download the full dataset or full GloVe resource at startup. Generate a lightweight reports archive, a four-model experiment archive, and a trained application archive. Hosting/publication is a separate step.

## Dependency reliability and testing

Validate a coherent dependency set including Trainer/Accelerate and plotting. Avoid Project 1's install/restart loop: never require a restart before installing dependencies; never require factory reset to apply an installation; never leave a recurring process-kill cell in Run All. Prefer an isolated project runtime for preparation/training if notebook-preloaded packages conflict, while preserving compatible CUDA PyTorch access. Any unavoidable one-time kernel restart must be explicit and tested.

Execute the real source/embedding download and full preparation in the cloud where supported. Test mapping cases, class order, split/vocabulary integrity, pretrained rows, masking/padding invariance, actual gradients, attention normalization, clean reloads and both-model application export. Run tiny real neural checks for all four architectures using training/validation only. Verify scheduling overlap/isolation, failure handling, single-GPU fallback, archives and CPU inference.

Validate notebook schema, code, embedded-file identity, cell order and clean execution where possible. Default to full mode with bounded checks first; provide a smoke mode that neither evaluates the real test set nor claims meaningful trained metrics.

Clearly distinguish static inspection, dummy-process tests, CPU neural runs, native-kernel runs, CUDA execution and full training. Do not claim two-GPU Kaggle execution or full model quality was tested unless actually executed. List environment-blocked checks and the smallest workable next step; do not fabricate results.

Deliver the finished `.ipynb`, short exact Kaggle instructions and a verification summary. Complete implementation and available tests rather than returning only a plan. Routine design decisions are covered by the proposal; record audit-dependent settings and raise only genuine source/access or scope blockers.
