# Environment and data setup

This repository is an archival research-code collection. It does not provide a tested one-command training environment or the large assets required to reproduce every reported result.

## Lightweight archive inspection

Use Python 3.11 or newer to run:

```sh
python3 scripts/validate_archive.py
python3 scripts/render_results.py
```

No ML packages, datasets, or GPU are needed for these checks. For interactive reading of research notebooks, use JupyterLab or another notebook viewer.

## Research environments reported in the thesis

Appendix L reports separate university-cluster and Google Colab environments:

| Component | University cluster | Colab |
|---|---|---|
| Python | 3.11.15 | 3.12.13 |
| TensorFlow | 2.20.0 | 2.20.0 |
| Keras | 3.13.2 | 3.13.2 |
| NumPy | 2.4.2 | 2.0.2 |
| pandas | 3.0.2 | 2.2.2 |
| matplotlib | 3.10.8 | 3.10.0 |
| scikit-learn | 1.8.0 | 1.6.1 |
| scipy | 1.17.1 | Not listed |
| seaborn | Not listed | 0.13.2 |

These values are transcribed from the submitted thesis, **not a freshly resolved or tested dependency lockfile**. Other imports include `tensorflow_datasets`, `medmnist`, and Pillow. Version/platform compatibility must be checked for the environment and experiment being restored; the archive intentionally does not promise that installing one combined package list recreates both systems.

The thesis reports NVIDIA A40 GPUs for generative-model training and a Tesla T4 via Colab for the remaining experiments. Equivalent hardware and sufficient storage/memory may be needed. Full training was not rerun when preparing this public archive.

## Datasets

Obtain the datasets from their original providers and observe their licences and citation requirements:

- [Fashion-MNIST original project](https://github.com/zalandoresearch/fashion-mnist) and [TensorFlow Datasets loader](https://www.tensorflow.org/datasets/catalog/fashion_mnist).
- [EuroSAT original project](https://github.com/phelber/eurosat); the saved preprocessing notebook expects the RGB image directory to be prepared locally.
- [MedMNIST official code](https://github.com/MedMNIST/MedMNIST), including PathMNIST. The saved preprocessing code uses the MedMNIST interface.

These links identify sources; they are not proof that the current downloads reproduce the precise historical inputs. Restore the original split definitions and preprocessing choices from the code and thesis, rather than substituting new splits silently.

## Paths and execution order

1. Choose a separate working-data root, outside this source repository if possible.
2. Set `PROJECT_DIR`, `DATA_DIR`, and related paths in the selected notebook/script. Personal user/cluster prefixes have been replaced with `/path/to/thesis_project` or generic user placeholders. Original `/content/drive/MyDrive/thesis_project` and `/workspace` conventions may remain in Colab/RunPod variants; adapt them to your environment.
3. Start with EDA and `05_preprocessing.ipynb`, preparing the expected real train/validation/test arrays. Not all saved run variants expect the same layout: some generation scripts load `.npz` files directly from their configured data root, while classification notebooks expect `data/processed/seed_42/`.
4. Restore or retrain the selected generator, then prepare synthetic arrays with the exact keys expected downstream. Do not assume the presence of checkpoints or generated data in this repository.
5. For feature experiments, run the relevant MobileNetV2 extraction workflow and provide the feature arrays expected by the classifier.
6. Configure and execute the appropriate classification notebook against real held-out test arrays. Preserve the original split boundaries when making any environment adaptations.

Review each selected file before execution: many are long-running training workflows and create output directories or checkpoints. Intermediate arrays, weights, and plotting outputs remain outside Git by default. Restoring the entire workflow may require additional configuration or fixes; this archive does not fill gaps by fabricating missing original code.
