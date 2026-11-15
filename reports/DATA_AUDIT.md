# Real data audit

filtered, leakage-controlled six-class GoEmotions-derived task; not the unchanged published benchmark

          train  validation  test
joy       15044        1929  1858
sadness    2313         266   283
anger      4283         555   572
fear        537          72    80
surprise   3845         459   488
disgust     495          61    76

All source bytes and revisions are pinned. Vocabulary and N/(6*n_c) weights use training only. Train p99.5 >128 =>256, otherwise128; head truncation; audit residual truncation

Neutral-bearing and cross-emotion examples are excluded. Consistent duplicates prefer test, validation, then train. Conflicting connected groups are removed. NFC + whitespace collapse + casefold; punctuation retained; duplicate connected groups by source ID OR normalized full text

Surprise includes curiosity/confusion/realization under the published mapping. Scores concern an unambiguous subset of English Reddit comments; sarcasm, mixed emotions, domain shift, and rare-class uncertainty remain. Attention is an inspection aid; probabilities are uncalibrated.
