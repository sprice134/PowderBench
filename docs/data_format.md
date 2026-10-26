# PowderBench data formats

The [root README](../README.md) gives release counts and access instructions. All paths below are relative to the repository root. Original scene and image identifiers are preserved in directory names.

## Synthetic scenes

`PowderBench/synthetic/<powder>/scene_NNN/` contains one scene. Rendered and translated PNGs are 1024 × 768 pixels. The translated image is named `scene_NNN_HC.png`, `_MC.png`, or `_LC.png`; its filename records the contrast used.

| File | Meaning |
| --- | --- |
| `snapshot.png` | Rendered scene. |
| `scene_NNN_<contrast>.png` | Gemini-5 SEM-style translation. |
| `visible_instances.npz` | `visible_ids`: uint8 `(768,1024)` instance-ID image; 0 is background. |
| `amodal_masks.npz` | Bit-packed `masks` and `original_shape=(N,768,1024)`. |
| `depth.exr`, `depth.png` | Floating-point depth and display visualization. Treat the EXR values as scene depth; no physical calibration is asserted here. |
| `assembled_scene.obj` | Settled scene geometry, not the individual-particle input scans. |
| `normals.png`, `ao.png`, `instance.png` | Conditioning and visualization images. |
| `scene_meta.json`, `camera_data.json`, `generation_metadata.json` | Object-to-label mapping, camera, and generation settings. |

Decode amodal masks with:

```python
import numpy as np

with np.load("PowderBench/synthetic/ParrotMica/scene_000/amodal_masks.npz") as data:
    shape = tuple(data["original_shape"])
    masks = np.unpackbits(data["masks"], axis=-1)[:, :, :shape[2]]
    assert masks.shape == shape
```

`scene_meta.json` maps objects to visible IDs and amodal stack indices. Its `source_stl` values are historical references to external particle inputs; those files are not bundled. The dataset contains 537 complete scene directories across nine powders.

## Real SEM benchmark

`PowderBench/real_sem/<original-key>/` contains `original.png` and `final_masks.npz`. The NPZ stores boolean `masks` of shape `(N,H,W)` and matching `statuses`. `manifests/real_sem.jsonl` records the preserved key and its `train`, `val`, or `test` split. The fixed split has 214/53/67 images and 7,018 nonempty visible instances in total.

No experimental synthetic train/validation assignment is part of this release. Consumers may define their own synthetic split, but it should not be represented as the historical primary-run split; its saved identities were not recovered.

## Translation-fidelity audit

`PowderBench/translation_audit/<configuration>/<powder>/scene_NNN_<contrast>/` contains `original.png` and primary manual `final_masks.npz`, with the same mask format as the real benchmark. Configuration, powder, scene, and contrast are encoded in the path. The matching rendered visible mask is under `PowderBench/synthetic/<powder>/scene_NNN/visible_instances.npz`. There are 674 usable VLM pairs and 45 CycleGAN pairs; no second-reviewer labels are included.
