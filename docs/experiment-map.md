# Experiment map

Paths below are relative to `experiments/`. This is a navigation map of surviving files, not a certification that every listed version produced the final reported results.

## Data exploration and preprocessing

- `eda/`: four dataset and cross-dataset EDA notebooks. The Fashion-MNIST filename begins with a space in the original archive.
- `05_preprocessing.ipynb`: dataset loading, stratified splits, and processed `.npz` outputs.
- Seed 42 and data-path conventions recur in the saved experiments; confirm the parameters in the particular version you intend to execute.

## Conditional GAN experiments

| Dataset | Useful starting points |
|---|---|
| EuroSAT | `06_cgan_eurosat_seed42.py` and its notebook; root inspection notebooks |
| Fashion-MNIST | `CGAN_8(FMNIST)/cgan_fashion_mnist.py` and its notebook |
| PathMNIST | `CGAN_9(PMNIST)/cgan_path_mnist.py`, `cgan_path_mnist_final.py`, and corresponding notebooks |
| PathMNIST execution variants | `CGAN_PMNIST_FINALS/files/`, `CGAN_PMNIST_SKIP1/`, `CGAN_PMNIST_SKIP2/`, `CGAN_PMNIST_SKIP3/`, `CGAN_PM_RUNPOD/` |

The preserved implementations include critic/generator code and gradient-penalty training. The `CGAN_1(kullanılmayacak)` and `CGAN_2(STARTFROM-İPTAL)` names explicitly indicate unused/cancelled trials; `CGAN_3` through `CGAN_7` contain additional explored versions. They are retained for research history, not recommended as authoritative final configurations.

## Diffusion experiments

| Dataset | Surviving implementations |
|---|---|
| EuroSAT | Root `07_ddim_eurosat.ipynb`, `DDIM_1/07_ddim_eurosat.ipynb`, and `DDIM_2/07_ddim_eurosat.ipynb` |
| Fashion-MNIST | `DDIM_3(FMNIST)/08_ddim_fashion_mnist_seed42.py` and notebook |
| PathMNIST | `DDIM_4(PMNIST)/09_ddim_path_mnist_seed42.py` and notebook |

Inspection notebooks accompany the generation code. Compare schedule, architecture, training parameters, and result paths with the thesis before selecting a version; copies are not silently merged into an invented single “final” implementation.

## Classification

`CLASSIFICATION/` contains 40 notebooks:

- `11a`–`11j`: CNN hyperparameter search and real/synthetic pixel-image experiments.
- `12a`–`12j`: random forest search and real/synthetic pixel-image experiments.
- `16a`–`16j`: CNN-head search and real/synthetic feature experiments.
- `17a`–`17j`: feature-based random forest search and experiments.

These files contain 36 dataset/model/input evaluation conditions plus four hyperparameter-search notebooks. Earlier or duplicate copies remain at the root and inside `CNN_*`, `RF_*`, `FE_CNN_*`, and `FE_RF_*` directories. The shared naming does not establish byte-for-byte equivalence or final version ordering.

## Feature extraction and analysis

- `FE_FE/` and root `13_feature_extraction_real.ipynb`, `14_feature_extraction_ddim.ipynb`, `15_feature_extraction_cgan.ipynb`: frozen MobileNetV2 feature extraction.
- `CENTROID/16_feature_centroid_distance.ipynb` and `CENTROID/Untitled.ipynb`: additional feature-space analysis.
- Other `inspection` notebooks: generation quality and sample inspection. Cleared image outputs can be regenerated only after restoring the required artifacts and environment.

## Inventory

[`source-manifest.json`](source-manifest.json) lists all 149 main research files plus 33 recovered autosaves, source-relative paths, original code hashes, and publication-copy hashes. It does not disclose personal machine paths. Notebooks were sanitized for publication; source hashes and published-file hashes serve different purposes and are not expected to match.

## Recovered autosaves

See [`archive/recovered-autosaves/README.md`](../archive/recovered-autosaves/README.md) for 33 additional code snapshots, including a class-wise results notebook whose regular counterpart was not found. Autosaves are retained separately and are not assumed to be final versions.
