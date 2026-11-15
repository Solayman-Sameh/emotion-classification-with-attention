# Frozen held-out error analysis

Selection remains distilbert. These outcomes do not trigger retuning or winner replacement.


## distilbert

fear: support=80, precision=0.6036, recall=0.8375, F1=0.7016. Interpret with the reported small support.

disgust: support=76, precision=0.6026, recall=0.6184, F1=0.6104. Interpret with the reported small support.

Largest confusions (true → predicted): joy → anger: 91, joy → surprise: 88, anger → surprise: 49, anger → sadness: 49, surprise → joy: 47.

negation: 870 records, accuracy 0.7402, six-label macro F1 0.6801.

has_vocab_oov: 986 records, accuracy 0.7769, six-label macro F1 0.6917.

recurrent_truncated: 0 records; no observed examples (or not applicable to this architecture).

transformer_truncated: 0 records; no observed examples (or not applicable to this architecture).

These are descriptive slices, not causal effects; their class mixtures differ. Inspect errors.jsonl for wording, ambiguity and possible sarcasm rather than assigning those explanations automatically.


## lstm

fear: support=80, precision=0.6279, recall=0.6750, F1=0.6506. Interpret with the reported small support.

disgust: support=76, precision=0.4182, recall=0.6053, F1=0.4946. Interpret with the reported small support.

Largest confusions (true → predicted): joy → surprise: 177, joy → anger: 169, anger → surprise: 103, anger → joy: 62, joy → sadness: 51.

negation: 870 records, accuracy 0.5989, six-label macro F1 0.5541.

has_vocab_oov: 986 records, accuracy 0.6755, six-label macro F1 0.5729.

recurrent_truncated: 0 records; no observed examples (or not applicable to this architecture).

transformer_truncated: 0 records; no observed examples (or not applicable to this architecture).

These are descriptive slices, not causal effects; their class mixtures differ. Inspect errors.jsonl for wording, ambiguity and possible sarcasm rather than assigning those explanations automatically.


## gru

fear: support=80, precision=0.6154, recall=0.7000, F1=0.6550. Interpret with the reported small support.

disgust: support=76, precision=0.4087, recall=0.6184, F1=0.4921. Interpret with the reported small support.

Largest confusions (true → predicted): joy → surprise: 163, joy → anger: 134, anger → surprise: 88, anger → joy: 72, surprise → joy: 51.

negation: 870 records, accuracy 0.6310, six-label macro F1 0.5746.

has_vocab_oov: 986 records, accuracy 0.7028, six-label macro F1 0.6137.

recurrent_truncated: 0 records; no observed examples (or not applicable to this architecture).

transformer_truncated: 0 records; no observed examples (or not applicable to this architecture).

These are descriptive slices, not causal effects; their class mixtures differ. Inspect errors.jsonl for wording, ambiguity and possible sarcasm rather than assigning those explanations automatically.


## bilstm_attention

fear: support=80, precision=0.6413, recall=0.7375, F1=0.6860. Interpret with the reported small support.

disgust: support=76, precision=0.4048, recall=0.6711, F1=0.5050. Interpret with the reported small support.

Largest confusions (true → predicted): joy → anger: 139, joy → surprise: 120, anger → surprise: 68, anger → joy: 63, surprise → joy: 60.

negation: 870 records, accuracy 0.6356, six-label macro F1 0.5644.

has_vocab_oov: 986 records, accuracy 0.7160, six-label macro F1 0.6320.

recurrent_truncated: 0 records; no observed examples (or not applicable to this architecture).

transformer_truncated: 0 records; no observed examples (or not applicable to this architecture).

These are descriptive slices, not causal effects; their class mixtures differ. Inspect errors.jsonl for wording, ambiguity and possible sarcasm rather than assigning those explanations automatically.


The single-label filter excludes neutral-bearing and cross-emotion comments. Performance applies to the retained subset; the app still forces neutral/mixed text into six labels. Surprise includes curiosity/confusion/realization. Same-class encoding collisions are disclosed in data_audit.json; zero full-text overlap does not imply zero encoded overlap. No calibration or causal attention claim is made.