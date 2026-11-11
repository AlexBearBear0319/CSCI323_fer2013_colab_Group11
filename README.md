# FT11: Facial expression recognition on FER2013

Code and saved results for our CSCI323 Modern Artificial Intelligence group project (University of Wollongong, Session 4 2026). The report compares four networks on FER2013 and then tests two ways of improving the smaller ones: focal loss and data augmentation.

All scores in the report and in this repository are **validation** scores (seed 42, one run per setting). The 7,178 source test images were never scored.

## What is in this repository

| Notebook (`notebooks/`) | What it does | Report sections | Written and run by |
|---|---|---|---|
| `01_Baselines_MLP_SmallCNN_ResNet18_PaperCNN.ipynb` | Dataset audit and split; trains the MLP, Small CNN, ResNet18 and the paper-based CNN (adapted from Singh & Nasoz, 2020); comparison tables, figures, cost measurements | 3.1 to 3.4, 3.7, 3.8 | Alex |
| `02_FocalLoss_SmallCNN.ipynb` | Small CNN with cross-entropy and focal loss (gamma 1, 2, 4) | 3.5.1, 3.5.2 | Harini |
| `03_FocalLoss_MLP.ipynb` | MLP with cross-entropy and focal loss (gamma 1, 2, 4), plus the combined MLP and Small CNN comparison | 3.5.1, 3.5.3 | Abinaya |
| `04_Augmentation_SmallCNN.ipynb` | Small CNN with ten augmentation settings (flip, rotation, shift, zoom, brightness, contrast, crop, random erasing) | 2.10, 3.6 | Natalie |

Saved outputs of the runs are in `results/`:

| Folder | Contents |
|---|---|
| `results/baselines_v6/` | Overall and per-class tables, confusion matrices, learning curves, figures, the split manifest, and one folder per model under `models/` (config, training history, validation predictions, metrics) |
| `results/focal_loss_small_cnn/` | Four Small CNN runs: comparison and per-class CSVs, per-run folders |
| `results/focal_loss_mlp/` | Four MLP runs, the combined MLP and Small CNN table, and `combined_macro_f1.png` |
| `results/augmentation/` | Ten augmentation runs: `augmentation_results.csv`, figures, per-run folders |

## Data

The dataset is **not** stored here. Every notebook downloads it with `kagglehub` (`msambare/fer2013`, 48 × 48 grayscale images in seven classes). The notebooks audit it for exact duplicates, keep duplicate groups together, and hold out 15% of the training folder for validation (seed 42), which gives 23,743 training, 4,190 validation and 7,178 test images, with 776 images excluded. The exact split is recorded per image in `split_manifest.json`.

The Kaggle download was not pinned to a release, so the data are identified by the SHA-256 fingerprints saved in `source_provenance.json` and `study_config.json`. See the dataset page on Kaggle for its terms of use.

## How to run

1. Open a notebook in Google Colab (a GPU runtime is optional) or in a local Jupyter session. Python 3.10 or later; `pip install -r requirements.txt` installs what the notebooks import.
2. Run the cells from top to bottom. The first cells install missing packages, select the device (CUDA, MPS or CPU), download FER2013 and build the split.
3. Training starts after the split. Results are written to a run folder and summarised in tables and figures in the same notebook.

Shared settings in every notebook: Adam, learning rate 0.001, batch size 64, at most 20 epochs, early stopping with patience 3 on validation macro-F1, checkpoint chosen by validation macro-F1 (ties broken by lower validation loss), seed 42.

The baseline run used a CPU-only Colab runtime with one PyTorch thread, so it is slow: about 4.8 minutes for the MLP, 12.3 for the Small CNN, 77.7 for ResNet18 and 481.5 for the paper-based CNN. The focal-loss and augmentation notebooks record the device used for each run in their result tables. Training times from different devices are not comparable.

## Main results (validation, seed 42)

| Model | Accuracy (%) | Macro-F1 | Training time (min, CPU) |
|---|---:|---:|---:|
| ResNet18 (from scratch) | 54.42 | 0.5037 | 77.7 |
| Paper-based CNN | 51.24 | 0.4571 | 481.5 |
| Small CNN | 48.81 | 0.3920 | 12.3 |
| MLP | 36.37 | 0.2700 | 4.8 |

Source: `results/baselines_v6/overall_performance.csv`.

Focal loss with gamma 1, 2 and 4 lowered the Small CNN's macro-F1 (0.3920 with cross-entropy, 0.3840, 0.3733 and 0.3597 with focal loss) and moved the MLP's by only 0.004 to 0.005. Among the augmentation settings, horizontal flip scored highest (macro-F1 0.4023), but the gain is about the size of the run-to-run difference we measured (0.0100 between two runs with the same settings on different hardware). Details and caveats are in the report.

## What is not included

- The FER2013 images (downloaded by the notebooks).
- The ResNet18 checkpoint (`best_checkpoint.pt`, 44.7 MB), left out for size. Its configuration, training history, validation predictions and metrics are in `results/baselines_v6/models/resnet18__baseline__seed42/`. Because of this one missing file, `sha256sum -c checksums.sha256` in `results/baselines_v6/` reports that file as missing.
- Test-set scores, because the test set was not scored.

## Limits

One seed per setting, so repeatability was not measured. Baseline and Small CNN runs used a CPU; the MLP and augmentation runs used a CUDA GPU in Colab. No class-weighted loss, transfer learning, larger inputs or learning-rate schedule was tested.

## References

- Singh, S., & Nasoz, F. (2020). Facial expression recognition with convolutional neural networks. 2020 10th Annual Computing and Communication Workshop and Conference (CCWC). https://doi.org/10.1109/CCWC47524.2020.9031283
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- FER2013 (image-folder version): https://www.kaggle.com/datasets/msambare/fer2013
