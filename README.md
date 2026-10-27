# PowderBench

**PowderBench** is a dataset of dense powder scenes for scanning electron microscopy (SEM) characterization. **PowderGen** is the accompanying scene-generation and image-translation pipeline. They accompany *PowderGen: Physics-Based Data Generation for Multi-Modal SEM Powder Characterization* ([manuscript PDF](paper/WACV_Price_2027_Camera_Ready__Unsubmitted_.pdf)).

PowderBench contains **537 complete synthetic scenes** across nine powders, **334 hand-labeled real SEM images** with 7,018 visible particle instances, and **719 hand-labeled translation-fidelity audit images**. The released synthetic scenes include renders, SEM-style translations, visible and amodal masks, depth, and assembled 3D geometry. Individual-particle scans are not included.

## Get the data

The dataset is included under `PowderBench/`. A full clone includes the dataset bytes as well as PowderGen. The `PowderGen/` code can be used to create new scenes with separately acquired particle STL inputs.

## Dataset layout

| Path | Content |
| --- | --- |
| `PowderBench/synthetic/<powder>/scene_NNN/` | Rendered and Gemini-5 translated images, visible/amodal instance masks, depth, assembled OBJ geometry, and scene metadata. |
| `PowderBench/real_sem/<original-key>/` | `original.png` and primary manual `final_masks.npz`; fixed train/val/test identities: **214/53/67**. |
| `PowderBench/translation_audit/<configuration>/<powder>/scene_NNN_<contrast>/` | Translated image and primary manual mask: **674 VLM + 45 CycleGAN** usable pairs. |
| `manifests/real_sem.jsonl` | Fixed real benchmark train/validation/test assignments. |
| `manifests/translation_provenance.json` | Model IDs and translation conditioning settings; prompt text is in `PowderGen/translation/prompts/`. |

For one synthetic record:

```python
from pathlib import Path
import numpy as np
from PIL import Image

scene = Path("PowderBench/synthetic/ParrotMica/scene_000")
render = Image.open(scene / "snapshot.png")
with np.load(scene / "visible_instances.npz") as labels:
    visible_ids = labels["visible_ids"]  # 0 is background; shape (768, 1024)
```

See [data formats](docs/data_format.md) for mask decoding and modality details. The fixed **real SEM benchmark** split is 214 train, 53 validation, and 67 test images. No experimental synthetic train/validation split is prescribed for the dataset release.

## PowderGen

[PowderGen instructions](docs/powdergen.md) cover input preparation, separate Blender and translation environments, one-scene commands, and full-generation commands. Processed particle STL inputs are not included. The saved translation [prompts and model IDs](manifests/translation_provenance.json) are preserved. Existing-input STL import and translation-conditioning preparation passed; full scene generation and model inference were not rerun for this release.

## Sources and citation

Particle geometry was prepared from [PARROT](https://doi.org/10.1017/S143192762101391X) and [The Sand Atlas](https://doi.org/10.1002/gdj3.70008). SEM-style translations and audit images were produced with Gemini, GPT Image 1 (`gpt-image-1`), and Qwen Image Edit (`Qwen/Qwen-Image-Edit-2511`); the audit also includes a CycleGAN baseline. The exact model IDs, input configurations, and prompts are in the [translation provenance](manifests/translation_provenance.json). Cite the PowderGen manuscript and the source datasets when using the release. A dataset-record citation will be added after publication.

```bibtex
@unpublished{price2027powdergen,
  title={PowderGen: Physics-Based Data Generation for Multi-Modal SEM Powder Characterization},
  author={Price, Stephen and Cote, Danielle L. and Rundensteiner, Elke A.},
  note={WACV 2027 manuscript}
}

@article{ditscherlein2022parrot,
  title={PARROT: A Pilot Study on the Open Access Provision of Particle-Discrete Tomographic Datasets},
  author={Ditscherlein, Ralf and Furat, Orkun and L{"o}wer, Erik and Mehnert, Raik and Trunk, Robin and Lei{ss}ner, Thomas and Krause, Mathias J. and Schmidt, Volker and Peuker, Urs A.},
  journal={Microscopy and Microanalysis},
  year={2022},
  doi={10.1017/S143192762101391X}
}

@article{vego2025sandatlas,
  title={The Sand Atlas},
  author={Vego, Ilija and Marks, Benjy},
  journal={Geoscience Data Journal},
  volume={12},
  pages={e70008},
  year={2025},
  doi={10.1002/gdj3.70008}
}
```

Questions and corrections can be raised through this repository's GitHub Issues after publication.
