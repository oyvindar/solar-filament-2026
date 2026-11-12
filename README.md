# Solar Filament Segmentation Challenge 2026: metric-optimal instance selection

Instance segmentation of solar filaments in GONG H-alpha full-disk images (2048 x 2048) for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup 2026).

**Idea in one paragraph.** The score is Panoptic Quality (PQ) pooled over every (annotator, image)
pair. For pixel-disjoint predictions the PQ denominator is
`0.5 * (R * #predictions + #GT)`, independent of which predictions match. Each predicted
filament therefore contributes on its own, and keeping it pays off exactly when its expected
matched IoU (`P(IoU > 0.5) * E[IoU | match]`, averaged over annotators) exceeds `PQ / 2`. We
(1) train U-Nets to predict the probability that a random annotator marks a pixel,
(2) enumerate candidate instances as the component tree of that map over 15 thresholds,
(3) learn the expected matched IoU of every candidate, and (4) choose the best set of
non-overlapping candidates per image exactly, with a tree dynamic program and the global
threshold `tau = PQ / 2` (Dinkelbach).

## Results (5-fold, date-grouped cross-validation on all 707 training images)

| Post-processing on U-Net (ResNet34) maps | Out-of-fold PQ | Public LB |
|---|---|---|
| Connected components at a fixed threshold (0.35, >= 500 px) | 0.411 | 0.35 |
| Component tree + learned expected IoU + exact selection (tau = 0.28) | **0.429** | **0.37** |
| Same selection with perfect IoU estimates (upper bound of the candidate set) | 0.55 | - |
| Reference: one human annotator scored against the others (296 images) | 0.340 | - |

The learned selection beats the fixed threshold in every fold (+0.011 to +0.024). Matched masks
agree with annotators better than annotators agree with each other (SQ 0.680 vs 0.634; RQ 0.631
vs 0.536). Per-fold numbers: `results/cv_scores_per_fold.csv`.

## Data use

Only the competition's training images and annotations are used for training, plus ImageNet
weights for the encoders (timm). No external labels (MAGFiLO 1.0 release, SWEFil, ...) are used;
inference uses only the given test image. Folds group images by observation date, so the same
day seen from different GONG sites never appears in both training and validation.

## Pipeline

| Stage | Where | Code |
|---|---|---|
| 1. Soft-label U-Net per fold (ResNet34 and ResNet50 encoders, 2 channels: intensity and log flat-field), D4 test-time augmentation, out-of-fold and test probability maps | Kaggle GPU (2 x T4) | `kaggle/train_unet_template.py`, `kaggle/make_kernels.py` |
| 2. Component tree over 15 thresholds; shape, probability, contrast and tree-context features; matched-IoU labels per annotator | CPU | `src/fsc/candidates.py`, `src/fsc/extract.py` |
| 3. Hurdle scorer: P(any annotator matched) x E[matched IoU], LightGBM, cross-fitted by fold | CPU | `src/fsc/scorer.py` |
| 4. Exact max-weight antichain per image with gain `u_hat - tau` | CPU | `src/fsc/select.py`, `src/fsc/predict.py` |
| 5. Compressed COCO RLE, format checks (no empty masks, no shared pixels) | CPU | `src/fsc/metric.py`, `scripts/make_submission.py` |

The PQ implementation (`src/fsc/metric.py`) reproduces the organisers'
[self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
to within 1e-9.

## Reproduction

```
pip install -r requirements.txt
python kaggle/make_kernels.py --variant=r34 0 1 2 3 4     # one Kaggle kernel per fold
kaggle kernels push -p kaggle/kernels/fsc26-unet-r34-f0 --accelerator NvidiaTeslaT4   # ... per fold
python kaggle/fetch_outputs.py work/kaggle_out saudsaudesen/fsc26-unet-r34-f0 ...
python scripts/make_submission.py --maps work/kaggle_out/fsc26-unet-r34-f* --scorer work/scorer_hurdle_all.pkl --tau 0.28 --out submission.csv
```

`notebooks/end_to_end.ipynb` runs the whole pipeline in one Kaggle session.

## Negative results kept for the record

* A finer tree (33 thresholds) raises the upper bound (0.555 -> 0.563) but not the learned selection (0.4324 vs 0.4320).
* LightGBM capacity changes (31 or 127 leaves) do not help (-0.002).
* EfficientNet-B3 encoders crash on Kaggle's T4 image (torch 2.11, fp16 depthwise convolution: CUDA misaligned address).

## License

Code: MIT. The competition data are not redistributed here.
