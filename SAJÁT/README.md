# YOLO26 spike detection

This project detects and classifies neuronal spikes with YOLO26n. Each image is a window of 128 channels × 128 samples (6.4 ms) of a filtered multichannel recording, and each sorted spike is a box labelled with its unit.

**Data:** rat neocortex recordings from the Ulbert Lab (128-site silicon probe, 20 kHz, Kilosort2 spike sorting), such as `Rat01_Insertion1_Depth2.nwb` (91 units) and `Rat01_Insertion2_Depth2.nwb` (100 units). The NWB files and the `DATASETS/` folder sit two levels above this folder, in `yolo26/`.

| File | Purpose |
| --- | --- |
| `nwb_yolo_dataset.py` | Builds YOLO datasets from an NWB recording |
| `nwb_yolo_dataset_steps.ipynb` | Runs the builder one step at a time, with checks and plots |
| `train_yolo.py` | Trains YOLO26n on one stage and logs to Weights & Biases |
| `spikesorting_requirements.txt` | Package versions used for the builds and training |

## Setup

```bash
pip install -r SAJÁT/spikesorting_requirements.txt
```

On Windows with an NVIDIA GPU, install `torch` from https://pytorch.org first; the default pip build is CPU-only.

## 1. Build a dataset

```bash
cd SAJÁT
python nwb_yolo_dataset.py --nwb ../../Rat01_Insertion1_Depth2.nwb  # stage 3, RGB
python nwb_yolo_dataset.py --stages 1 2 3 --image-mode gray rgb     # all stages, both encodings
python nwb_yolo_dataset.py --quick                                  # small test build
```

The first build of a recording writes a preprocessed copy of the signal (12–19 GB) to `DATASETS/nwb_pp_cache/`, and later builds reuse it. Each build is written to its own `build_<time>_<mode>` folder.

| Stage | Windows | Contains |
| --- | --- | --- |
| 1 | `stage1_single_spike` | Exactly one labelled spike |
| 2 | `stage2_up_to_3_spikes` | Stage 1, plus windows with 2–3 spikes |
| 3 | `stage3_any_spike_count` | Stage 2, plus sliding windows with any number of spikes |

Every stage also has 10% empty windows, and the recording is split by time: 70% train, 15% validation, 15% test.

| Image mode | Channels |
| --- | --- |
| `gray` | Filtered signal on a fixed asinh brightness curve |
| `rgb` | R = gray, G = locally whitened signal, B = Teager-Kaiser (NEO) energy |

## 2. Train

Set `BUILD`, `STAGE`, `SINGLE_CLS`, `EPOCHS` and `WEIGHTS` at the top of `train_yolo.py`, then run:

```bash
python train_yolo.py
```

## Results so far

Stage 3 of `Rat01_Insertion1_Depth2` (91 units), validation scores from training:

| Image mode | mAP50 | mAP50-95 |
| --- | --- | --- |
| Gray (minimum box half-height 2) | 0.857 | 0.632 |
| RGB (no minimum box height) | 0.833 | 0.624 |

The two runs used different box sizes, so they are not yet a fair comparison. On the held-out test split, the RGB model found 88% of spikes within 3 samples and named the right unit for 97% of those it found.
