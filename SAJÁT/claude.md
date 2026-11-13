
```
# Review task: nwb_yolo_dataset.py — evaluate proposed changes

Context: `nwb_yolo_dataset.py` builds staged YOLO26 spike-detection datasets (128 ch × 128 samples)
from `Rat01_Insertion2_Depth2.nwb` + its sorting, via a preprocessed int16 cache. A review raised the
points below. For each one: check whether it's actually true in the code/data, run the suggested
check where possible (use the cache, keep runs small), and give a verdict: **fix now / later / not
needed**, with effort estimate and any risk to existing builds. Don't implement yet — report first.

## Correctness / label quality (highest priority)
1. **Stored templates vs images.** Boxes come from `mean_waveform_all_channels_filt`, assumed trough at
   index 40, same filter/units as our pipeline. Check: recompute mean templates for ~20–30 units from the
   cache (train split) and compare to stored ones: peak lag, amplitude ratio, correlation. Also print a
   histogram of `center_dt` — many units at ±5 = misalignment.
2. **Weak spikes labelled.** Every sorted spike is now labelled, even below 5σ. Check `weak_labels` share
   in stats.json. Option: config `weak_policy = label | background | drop_image`. Also consider measuring
   visibility as max over box rows instead of fixed `peak_row` (drift robustness).
3. **Duplicate sorter spikes.** Count spike pairs from different units within ~5 samples with heavily
   overlapping boxes. If common, dedupe (keep higher-amplitude unit).
4. **Symmetric boxes + edge clipping.** Symmetric boxes are wider than the waveform (more overlap → NMS may
   suppress close true spikes); edge-clipped boxes lose centre = spike time. Options: restore a
   min-visible-fraction rule; or tight boxes + YOLO pose keypoint at the trough. Assess effort.
5. **Filter parity with deployment.** Training uses zero-phase filtfilt (needs ≥50 ms look-ahead online).
   Evaluate adding a causal (`sosfilt`) cache variant if real-time Edge TPU deployment needs it.

## Robustness
6. Cache equality check: test several locations incl. ones spanning the 10 s write seams; replace `assert`
   with `raise`.
7. Label y is divided by W (works only because channels == window) — use separate H.
8. `anchor_margin_before` comment says margin precedes the box; code uses max(), not sum.

## Yield of stages 1–2
9. Anchor column is drawn once; try all columns within ±jitter and pick a random valid one.
10. Caps (`anchors_per_group_per_chunk`, `max_per_unit_count`, `stage3_max_images`) — make `None` = uncapped
    (stage-3 arithmetic in `pass2_tasks`/`select_items` needs a None guard). Keep val/test natural; balance
    train by sampling weights instead. Add per-reason anchor rejection counters to stats.json.

## Simplifications
11. Dead/duplicate code: `to_image` unused; `load_recording` duplicates `preprocess_chain`;
    `template_source` constant.
12. `STAGES` (min,max) values unused — drive stage filters from it instead of hard-coded `n_labels == 1`.
13. Replace hard links with one image+label pool and per-stage `.txt` image lists in data.yaml
    (labels depend only on window start).
14. Single LUT source (`ctx["lut"]` vs `enc["lut"]` can disagree in multi-mode builds); split `worker_state`.
15. `rgb_feat` B channel: consider k-lag smoothed TEO instead of 1-lag NEO.

## Output
A table: item | confirmed? | evidence (numbers/plots) | verdict | effort. Then a proposed order of changes.
```