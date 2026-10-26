# Running PowderGen

PowderGen has separate stages for processed STL input checks, Blender physics/rendering and labels, and translation conditioning/inference. Source single-particle scans and processed STL inputs are not included. Obtain PARROT scans from [PARROT](https://parrot.tu-freiberg.de/) and Sand Atlas scans from [The Sand Atlas](https://sand-atlas.scigem.com/), then provide processed STL files as `<mesh-root>/<powder>/*.stl`. The input preparation tool checks this layout; it does not create missing scans.

## Environments

Scene generation runs inside Blender with its own NumPy. The existing-input STL import preflight passed using Blender 5.0; a full one-scene physics/rendering run has not been tested for this release. Translation inference needs a separate environment:

```bash
python3 -m venv .venv-translation
. .venv-translation/bin/activate
python -m pip install -r PowderGen/requirements-translation.txt
```

Qwen additionally needs compatible PyTorch/CUDA, `diffusers`, and `transformers`. Gemini and OpenAI credentials are read from `GEMINI_API_KEY` and `OPENAI_API_KEY` environment variables. Keep credentials out of the repository.

## One-scene commands

Set `MESH_ROOT` to your processed STL tree. These commands write into ignored `generated/`, not the released dataset:

```bash
python PowderGen/prepare_inputs.py --mesh-root "$MESH_ROOT" --powder ParrotMica
blender -b -P PowderGen/blender/generate.py -- --mesh-root "$MESH_ROOT" --output-root generated/scenes --dataset ParrotMica --start_id 0 --num_scenes 1 --device CPU --min_faces 25000 --voxel_remesh_size 0.04 --voxel_smoothing 0.03 --voxel_smoothing_iterations 4
python PowderGen/translation/infer.py --dataset-dir generated/scenes --powder ParrotMica --output-root generated/translations --provider gemini-5 --start-index 0 --max-scenes 1 --available-options HC
```

The Blender command runs physics and rendering; translation inference calls a model/API. Neither was rerun to fill historical release gaps. For lower-cost preflights using existing inputs:

```bash
blender -b -P PowderGen/blender/generate.py -- --mesh-root "$MESH_ROOT" --output-root generated/preflight --dataset ParrotMica --preflight-only
python PowderGen/translation/infer.py --dataset-dir PowderBench/synthetic --powder ParrotMica --output-root generated/conditioning --provider gemini-5 --start-index 0 --max-scenes 1 --available-options HC --prepare-only
```

The STL import and five-input translation-conditioning stages passed these preflights. Conditioning order is render, instance overlay, normals, ambient occlusion, depth visualization. The 15 preserved prompts, exact model IDs, and settings are documented in [`manifests/translation_provenance.json`](../manifests/translation_provenance.json); the historical OpenAI model is `gpt-image-1`.

## Full generation commands

These are commands for a **new** run, with substantial simulation and translation costs. They were not run for release assembly:

```bash
for powder in ParrotAl2O3 ParrotDolomite ParrotLimestone ParrotMica ParrotQuartz ParrotSodalimeGlass SandAtlasHamburg SandAtlasOttawa_1micron_30mesh SandAtlasOttawa_7.3micron_30mesh; do
  blender -b -P PowderGen/blender/generate.py -- --mesh-root "$MESH_ROOT" --output-root generated/scenes --dataset "$powder" --start_id 0 --num_scenes 60 --device GPU --min_faces 25000 --voxel_remesh_size 0.04 --voxel_smoothing 0.03 --voxel_smoothing_iterations 4
done
python PowderGen/translation/infer.py --dataset-dir generated/scenes --output-root generated/translations --provider gemini-5 --start-index 0 --max-scenes 20 --available-options HC
python PowderGen/translation/infer.py --dataset-dir generated/scenes --output-root generated/translations --provider gemini-5 --start-index 20 --max-scenes 20 --available-options MC
python PowderGen/translation/infer.py --dataset-dir generated/scenes --output-root generated/translations --provider gemini-5 --start-index 40 --max-scenes 20 --available-options LC
```

These commands express the intended 20/20/20 contrast allocation for new scenes. The historical contrast of each released scene is recorded in its translated image filename.
