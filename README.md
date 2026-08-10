# Modified Double U-Net for BUSI

This project provides a simple three-class BUSI segmentation workflow:

- `0`: background / normal
- `1`: benign lesion
- `2`: malignant lesion

The corrected implementation keeps normal scans, uses explicit manifest labels, merges masks, prevents duplicate-content leakage, and supports PyTorch on NVIDIA CUDA or AMD ROCm.

## Radeon Cloud setup

Create a PyTorch + ROCm instance with persistent storage, open Notebook, then Terminal, and run:

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/Jonathan-Jesni/Modified_DoubleUNet_Implementation.git
cd Modified_DoubleUNet_Implementation

# Keep the ROCm torch and torchvision supplied by the cloud image.
python -m pip install -r requirements.txt

rocm-smi
python -c "import torch; assert torch.cuda.is_available(), 'GPU not detected'; assert torch.version.hip, 'ROCm PyTorch not detected'; print(torch.cuda.get_device_name(0), torch.version.hip)"
```

Do not install torch or torchvision separately over a working ROCm environment.

## Dataset

Copy the complete generated folder into the repository root:

```text
dataset_seg_BUSI_v2/
  manifest.csv
  images/
  masks/
  review/
  quarantine/
  dataset_metadata.json
```

The loader reads the immutable manifest; diagnosis is never inferred from filenames.
The final dataset has been fully hash-checked, so it does not need to be regenerated.

## Train, test, and predict

```bash
python train_BUSI.py
python test_BUSI.py
python predict_BUSI.py
```

- Training checkpoints and logs are saved under `runs/BUSI/`.
- Testing selects the checkpoint/foreground threshold on calibration data and reports the held-out fold.
- Predictions, masks, probability maps, overlays, and panels are saved under `files/predictions_BUSI/`.

The no-argument run uses the conservative v2 recipe: aspect-preserving 256-pixel
padding, ultrasound-safe augmentation, composite localization/class loss, staged
fine-tuning, and the identity-initialized adapter/ASPP repair.

Optional preflight before spending cloud credits:

```bash
python -m unittest discover -s tests -v
```

To resume an interrupted training run:

```bash
python train_BUSI.py --resume runs/BUSI/last.pt
```

Monitor the GPU in another terminal with `watch -n 2 rocm-smi`. Destroy the compute instance when training is finished; retain the PVC if you want to keep checkpoints and logs.

## Canonical BUSI v2 inventory

- 778 eligible images: 133 normal, 436 benign, and 209 malignant.
- 676 development images and 102 reserved confirmation images. The simple
  one-run command evaluates a held-out development fold and leaves the
  confirmation partition untouched.
- All normal masks are empty and all duplicate-content groups stay in one partition/fold.

CBIS scripts remain separate and are not used by the default BUSI workflow.
