# Solar Filament Segmentation Challenge 2026: selecting filaments by their expected matched IoU

Instance segmentation of solar filaments in GONG H-alpha full-disk images (2048 x 2048) for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup 2026). Author: Øyvind Andre Roth.

**Idea.** The score is Panoptic Quality (PQ) pooled over every (annotator, image) pair. For
pixel-disjoint predictions its denominator is `0.5 * (R * #predictions + #GT)` whichever predictions
match, so each predicted filament contributes on its own and keeping it pays off exactly when its
expected matched IoU, averaged over annotators, exceeds `PQ / 2` (Dinkelbach). The pipeline
(1) trains U-Nets to predict the probability that a random annotator marks a pixel, (2) lists
candidate filaments as the component tree of that map over 15 thresholds, learns every candidate's
expected matched IoU and picks the best non-overlapping set per image exactly with a tree dynamic
program, and (3) optionally re-segments every selected filament in a crop around it.

## Results

Out-of-fold PQ on all 707 training images (5 folds grouped by observation date), computed with the
organisers' scoring rule; public leaderboard scores from Kaggle.

| Method | PQ | SQ | RQ | Public LB |
|---|---|---|---|---|
| One annotator scored against the others (296 images) | 0.340 | 0.634 | 0.536 | - |
| Connected components at threshold 0.35, >= 500 px | 0.414 | | | 0.35 (ResNet34 maps) |
| Stage 2 on ResNet34 maps | 0.429 | | | 0.37 |
| Stage 2 on ResNet34 + ResNet50 maps | **0.434** | 0.683 | 0.635 | 0.37 |
| Stage 3: + crop refiner (threshold chosen on the other folds) | **0.438** | 0.681 | 0.643 | 0.37 |

* The learned selection beats the fixed threshold in every fold (+0.011 to +0.024 on ResNet34 maps);
  the refiner improves every fold (+0.002 to +0.006). Before training the refiner we had fixed an
  adoption threshold of +0.005; it reached +0.004 and is used only in the second final submission.
* Final Kaggle selections: the stage-2 ensemble (`sub_ens10_hurdle_t032.csv`) and its stage-3
  refinement (`sub_ens10_hurdle_t032_refined_t040.csv`). The notebook writes the identical files as
  `work/submission_stage2.csv` and `work/submission.csv` (checked row by row).
* The 0.06 gap between validation and the public leaderboard was tested and is not sampling noise
  (0.37 is ~5 standard deviations below 2,000 random 90-image validation subsets), not input shift
  (adversarial AUC 0.42-0.51), not an optimistic threshold (nested choice costs 0.002) and not the
  site/year mix (reweighted 0.431-0.434). Details: `results/step*.csv` and the report.

## Data use

Training uses only the competition's training images and annotations, plus ImageNet weights for the
encoders (timm). No external labels (the public MAGFiLO 1.0 release, SWEFil, ...) are used, and
inference uses only the given test image.

## Repository

| Path | Content |
|---|---|
| `notebooks/end_to_end.ipynb` | the whole pipeline in one notebook, with outputs (built by `notebooks/build_notebook.py`) |
| `src/fsc/metric.py` | PQ exactly as in the organisers' notebook (fast RLE version and a dense port), format checks, diagnostics |
| `src/fsc/candidates.py`, `extract.py` | component tree of a probability map, node features and matched-IoU labels |
| `src/fsc/scorer.py` | expected matched IoU: LightGBM hurdle model, cross-fitting |
| `src/fsc/select.py` | Dinkelbach threshold and exact max-weight antichain per image |
| `src/fsc/predict.py` | probability maps -> selected instances -> RLE |
| `src/fsc/refine.py`, `kaggle/refiner_template.py` | stage-3 crop refiner (training on Kaggle, pasting back locally) |
| `kaggle/train_unet_template.py`, `kaggle/make_kernels.py` | stage-1 U-Net training, one Kaggle kernel per fold and encoder |
| `kaggle/fetch_outputs.py`, `kaggle/run_queue.py` | download kernel outputs, queue runs under the 2-GPU-session limit |
| `scripts/make_submission.py`, `scripts/deep_eval_extract.py` | submission builder, raw-file extraction for the evaluation |
| `tests/test_pipeline.py` | unit tests: metric vs. organisers' rule and hand computation, tree DP, format checks |
| `data/folds.csv` | the fold of every training image |
| `results/` | out-of-fold result tables and the logs of every Kaggle training run |
| `report/` | 4-page report (ACM sigconf, organisers' template) and its figures |

## Reproduction

```
pip install -r requirements.txt
python -m pytest tests/                                 # or: python tests/test_pipeline.py
FSC_MODE=reuse jupyter nbconvert --execute notebooks/end_to_end.ipynb
```

`FSC_MODE=reuse` (default) reads the stage-1 probability maps of the published Kaggle runs
(`fsc26-unet-{r34,r50}-f{0..4}`; set `FSC_STAGE1` to their outputs) and recomputes stages 2 and 3 on
CPU in about an hour. `FSC_MODE=full` trains the ten U-Nets and the refiner first (about 8 h on
2 x T4, internet needed for the encoder weights). The data folder (`MAGFiLO_1.0_Kaggle_2026`) is found
under `FSC_INPUT` (default `/kaggle/input` on Kaggle, `../filament-segmentation-2026` locally).
Stage 1 alone runs as Kaggle kernels:

```
python kaggle/make_kernels.py --variant=r34 0 1 2 3 4
kaggle kernels push -p kaggle/kernels/fsc26-unet-r34-f0 --accelerator NvidiaTeslaT4   # one per fold
python kaggle/fetch_outputs.py work/kaggle_out <user>/fsc26-unet-r34-f0
```

## Negative results kept for the record

* A finer tree (33 thresholds) raises the upper bound of the candidate set (0.555 -> 0.563) but not
  the learned selection (0.4320 vs 0.4324 on two folds).
* LightGBM capacity changes (31 or 127 leaves) lower PQ by 0.002.
* EfficientNet-B3 encoders crash on the Kaggle T4 image (torch 2.11, half-precision depthwise
  convolution: CUDA misaligned address).
* The crop refiner missed its pre-set adoption threshold (+0.004 against +0.005).

## License

Code: MIT (`LICENSE`). The competition data are not redistributed here.
