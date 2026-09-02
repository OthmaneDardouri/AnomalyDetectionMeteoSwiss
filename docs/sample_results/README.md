# Sample results

Curated output from real MVTec AD runs, kept in version control as evidence
that the pipelines work end-to-end. These are **not** produced by the test
suite &mdash; `tests/` builds a tiny synthetic dataset and asserts on behaviour,
not on image quality.

Regenerate any of these yourself with (for example):

```bash
python train_test.py --model_name patchcore --product_class toothbrush --mode train \
    --train_path runs/train/patchcore_toothbrush
python train_test.py --model_name patchcore --product_class toothbrush --mode test \
    --train_path runs/train/patchcore_toothbrush --test_path runs/test/patchcore_toothbrush
```

| Folder | Model | Class | Contents |
| --- | --- | --- | --- |
| `patchcore_toothbrush/` | PatchCore | toothbrush | Original / ground-truth / predicted-mask comparisons, plus the AUC-ROC metrics file |
| `deep_feature_leather/` | Deep Feature AE | leather | ROC curve and normal-vs-anomaly score distribution |

New runs write to the git-ignored `runs/` directory; copy anything worth
keeping into this folder deliberately.
