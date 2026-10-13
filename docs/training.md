# Training on Kaggle

The detector is fine-tuned on a Kaggle GPU. Everything else (threshold tuning, evaluation, export, benchmarks) runs on the Mac. `docs/design.md` explains why.

`scripts/kaggle_train.sh` rebuilds the training data on Kaggle from the pinned OpenPII revision and the fixed augmentation seed, trains DistilBERT, and writes the model and logs to `/kaggle/working`. No data is uploaded.

## 1. Package the source

Commit first: the archive holds only committed files.

```
git archive --format=zip --prefix=edgeveil/ -o ~/Downloads/edgeveil-src.zip HEAD
```

## 2. Upload it as a private Kaggle dataset

On kaggle.com, open Datasets, choose New Dataset, upload `edgeveil-src.zip`, keep it private, and name it `edgeveil-src`. Kaggle extracts the zip.

## 3. Create the notebook

New Notebook, then in the right panel:

- Accelerator: GPU P100, or GPU T4 x2. The script uses one GPU either way (`CUDA_VISIBLE_DEVICES=0`), so the effective batch stays 32.
- Internet: on. This needs a phone-verified Kaggle account.
- Add Input: the `edgeveil-src` dataset.

Find the extracted folder:

```
!ls /kaggle/input/edgeveil-src
```

Then run the script with that folder, for example:

```
!bash /kaggle/input/edgeveil-src/edgeveil/scripts/kaggle_train.sh /kaggle/input/edgeveil-src/edgeveil
```

To run it without keeping the browser open, use Save Version, then Save & Run All (Commit). Kaggle sessions stop after 12 hours.

`EDGEVEIL_DEVICE_BATCH_SIZE` sets windows per device step (default 16, which fits a 16 GB GPU at 512 tokens). It must divide 32.

## 4. Bring the results back

From the notebook's Output tab, download:

- `distilbert-model.tar.gz`: the trained model and `training_run.json`
- `train.log`
- `prepare_report.json` and `augment_report.json`

Unpack the model:

```
tar -xzf ~/Downloads/distilbert-model.tar.gz -C models/
```

Check that Kaggle rebuilt the same data as the local pipeline. Both reports come from the same revision and seed, so they should match:

```
diff <(python3 -m json.tool ~/Downloads/augment_report.json) <(python3 -m json.tool data/openpii/augmented/report.json)
```

Then tune thresholds locally:

```
uv run edgeveil tune-thresholds --preset distilbert
```
