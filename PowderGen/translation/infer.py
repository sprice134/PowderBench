#!/usr/bin/env python3
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import argparse
import base64
import hashlib
import json
import random
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from google import genai
    from google.genai import types as gemini_types
except ImportError:
    genai = None
    gemini_types = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import torch
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    torch = None
    QwenImageEditPlusPipeline = None


NUM_SCENES = 10

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"

PROMPT_OPTIONS = (
    ("LC", PROMPT_DIR / "LowContrast5Image.txt"),
    ("MC", PROMPT_DIR / "MediumContrast5Image.txt"),
    ("HC", PROMPT_DIR / "HighContrast5Image.txt"),
)

GEMINI_MODEL_NAME = "gemini-3.1-flash-image-preview"
OPENAI_MODEL_NAME = "gpt-image-1"
QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

MAX_RETRIES = 3
RETRY_DELAY_SEC = 3.0
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768
OPENAI_REQUEST_SIZE = "1536x1024"
GLOBAL_MAX_DEPTH = 20.0
QWEN_SEED = 7
QWEN_NUM_INFERENCE_STEPS = 50
QWEN_TRUE_CFG_SCALE = 4.0
QWEN_GUIDANCE_SCALE = 1.0
QWEN_NEGATIVE_PROMPT = " "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate synthetic scenes with Gemini, OpenAI, or Qwen."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Root containing powder/scene_NNN folders")
    parser.add_argument("--powder", type=str, default=None,
                        help="Process only this powder folder")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory for translated output folders")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build five conditioning inputs locally without model initialization or inference")
    parser.add_argument(
        "--provider",
        choices=["gemini-5", "openai-5", "qwen-5"],
        default="gemini-5",
        help="Model provider to use.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=NUM_SCENES,
        help="Maximum number of scenes per dataset to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting scene index. For example, 5 starts at scene_005.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional RNG seed for prompt selection and Qwen generation.",
    )
    parser.add_argument(
        "--available-options",
        nargs="+",
        choices=[suffix for suffix, _ in PROMPT_OPTIONS],
        default=[suffix for suffix, _ in PROMPT_OPTIONS],
        help="Prompt suffixes allowed for random selection. Default: LC MC HC.",
    )
    parser.add_argument(
        "--qwen-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device for Qwen.",
    )
    parser.add_argument(
        "--qwen-offload",
        action="store_true",
        help="Enable CPU offload for Qwen on CUDA.",
    )
    return parser.parse_args()


def deterministic_color(instance_id):
    rng = np.random.default_rng(int(instance_id))
    color = rng.random(3)
    color = 0.35 + 0.65 * color
    return color


def label_boundary(labels):
    labels = np.asarray(labels)
    b = np.zeros_like(labels, dtype=bool)
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b &= (labels != 0)
    return b


def overlay_visible_instances(snapshot_rgb, visible_ids, alpha=0.45):
    snap = snapshot_rgb.astype(np.float32) / 255.0
    out = snap.copy()

    ids = np.unique(visible_ids)
    ids = ids[ids != 0]

    for instance_id in ids:
        mask = visible_ids == instance_id
        color = deterministic_color(instance_id)
        out[mask] = (1 - alpha) * out[mask] + alpha * color

    boundaries = label_boundary(visible_ids)
    out[boundaries] = np.array([1.0, 1.0, 1.0])

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def safe_open_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None

    try:
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as exc:
        print(f"Warning: failed to open image {path}: {exc}")
        return None


def load_text_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"Error: failed to read text file {path}: {exc}")
        return None


def resize_to_target(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return image.copy()
    return image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)


def get_or_generate_instance_mask(scene_dir: Path) -> Optional[Image.Image]:
    instance_path = scene_dir / "instance.png"
    if instance_path.exists():
        return safe_open_image(instance_path)

    npz_path = scene_dir / "visible_instances.npz"
    snap_path = scene_dir / "snapshot.png"

    if not npz_path.exists():
        print(f"Warning: missing visible_instances.npz in {scene_dir.name}")
        return None

    snapshot_img = safe_open_image(snap_path)
    if snapshot_img is None:
        print(f"Warning: missing or unreadable snapshot.png in {scene_dir.name}")
        return None

    try:
        with np.load(npz_path) as data:
            if "visible_ids" not in data:
                print(f"Warning: visible_instances.npz missing 'visible_ids' in {scene_dir.name}")
                return None
            visible_ids = data["visible_ids"]
    except Exception as exc:
        print(f"Warning: failed to read NPZ file {npz_path}: {exc}")
        return None

    snapshot = np.array(snapshot_img, dtype=np.uint8)
    overlay_img = overlay_visible_instances(snapshot, visible_ids)
    img = Image.fromarray(overlay_img)

    try:
        img.save(instance_path)
        print(f"Saved generated instance mask to {instance_path}")
    except Exception as exc:
        print(f"Warning: failed to save generated instance mask {instance_path}: {exc}")

    return img


def get_or_generate_depth_map(scene_dir: Path) -> Optional[Image.Image]:
    depth_png_path = scene_dir / "depth.png"
    if depth_png_path.exists():
        return safe_open_image(depth_png_path)

    depth_exr_path = scene_dir / "depth.exr"
    if not depth_exr_path.exists():
        print(f"Warning: missing depth.exr in {scene_dir.name}")
        return None

    depth_array = cv2.imread(str(depth_exr_path), cv2.IMREAD_UNCHANGED)
    if depth_array is None:
        print(f"Warning: failed to read depth.exr in {scene_dir.name}")
        return None

    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]

    depth_array = depth_array.astype(np.float32) / GLOBAL_MAX_DEPTH
    depth_array = np.clip(depth_array, 0.0, 1.0)
    depth_array = (depth_array * 255.0).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)
    depth_img = Image.fromarray(depth_rgb)

    try:
        depth_img.save(depth_png_path)
        print(f"Saved generated depth map to {depth_png_path}")
    except Exception as exc:
        print(f"Warning: failed to save generated depth map {depth_png_path}: {exc}")

    return depth_img


def collect_scene_inputs(scene_dir: Path) -> Optional[list[Image.Image]]:
    render_img = safe_open_image(scene_dir / "snapshot.png")
    if render_img is None:
        print(f"Warning: missing or unreadable Render image in {scene_dir.name}: snapshot.png")
        return None

    instance_img = get_or_generate_instance_mask(scene_dir)
    if instance_img is None:
        print(f"Warning: missing instance mask data in {scene_dir.name}")
        return None

    normal_img = safe_open_image(scene_dir / "normals.png")
    if normal_img is None:
        print(f"Warning: missing or unreadable Normal image in {scene_dir.name}: normals.png")
        return None

    ao_img = safe_open_image(scene_dir / "ao.png")
    if ao_img is None:
        print(f"Warning: missing or unreadable AO image in {scene_dir.name}: ao.png")
        return None

    depth_img = get_or_generate_depth_map(scene_dir)
    if depth_img is None:
        print(f"Warning: missing depth map data in {scene_dir.name}")
        return None

    return [render_img, instance_img, normal_img, ao_img, depth_img]


def prepare_scene_inputs(scene_dir: Path, output_root: Path, provider: str,
                         powder: str, prompt_suffix: str, prompt: str) -> Path:
    """Exercise the inference conditioning path in a scratch copy of one scene."""
    target = output_root / "prepared" / provider / powder / f"{scene_dir.name}_{prompt_suffix}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("snapshot.png", "visible_instances.npz", "normals.png", "ao.png", "depth.exr"):
        source, copied = scene_dir / name, target / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if copied.is_file():
            if hashlib.sha256(copied.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise ValueError(f"existing preparation input differs: {copied}")
        else:
            shutil.copyfile(source, copied)
    images = collect_scene_inputs(target)
    if images is None or len(images) != 5 or any(image.size != (TARGET_WIDTH, TARGET_HEIGHT) for image in images):
        raise ValueError(f"conditioning inputs are missing or dimensionally inconsistent: {scene_dir}")
    names = ("snapshot.png", "instance.png", "normals.png", "ao.png", "depth.png")
    report = {
        "source_scene": f"{powder}/{scene_dir.name}", "provider_configuration": provider,
        "prompt_suffix": prompt_suffix, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_order": list(names),
        "input_sha256": {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in names},
        "size": [TARGET_WIDTH, TARGET_HEIGHT], "inference_performed": False,
    }
    report_path = target / "conditioning.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def extract_first_gemini_image_from_response(response) -> Optional[Image.Image]:
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue

            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    img_bytes = BytesIO(inline_data.data)
                    with Image.open(img_bytes) as img:
                        return img.copy()
    except Exception as exc:
        print(f"Warning: failed to parse Gemini image from response: {exc}")

    return None


def debug_gemini_response(response) -> None:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        print("Debug: response had no candidates.")
        return

    for ci, candidate in enumerate(candidates):
        finish_reason = getattr(candidate, "finish_reason", None)
        print(f"Debug: candidate {ci} finish_reason={finish_reason}")

        content = getattr(candidate, "content", None)
        if content is None:
            print(f"Debug: candidate {ci} had no content.")
            continue

        parts = getattr(content, "parts", None) or []
        if not parts:
            print(f"Debug: candidate {ci} had no parts.")
            continue

        for pi, part in enumerate(parts):
            inline_data = getattr(part, "inline_data", None)
            text = getattr(part, "text", None)

            if inline_data and getattr(inline_data, "data", None):
                mime_type = getattr(inline_data, "mime_type", None)
                size_bytes = len(inline_data.data)
                print(
                    f"Debug: candidate {ci} part {pi} inline_data "
                    f"mime_type={mime_type} bytes={size_bytes}"
                )
            elif text:
                preview = text[:300].replace("\n", " ")
                print(f"Debug: candidate {ci} part {pi} text={preview!r}")
            else:
                print(f"Debug: candidate {ci} part {pi} had no inline_data or text.")


def choose_qwen_device(user_choice: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is not installed for Qwen.")

    if user_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"
    if user_choice == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_qwen_dtype(device: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed for Qwen.")

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_qwen_pipeline(device: str, dtype, offload: bool):
    if QwenImageEditPlusPipeline is None or torch is None:
        raise RuntimeError(
            "Qwen dependencies are missing. Install diffusers and torch to use --provider qwen."
        )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=dtype,
    )
    pipe.set_progress_bar_config(disable=False)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    else:
        pipe.to(device)

    return pipe


def sanitize_provider_name(provider: str) -> str:
    return provider.lower()


def choose_prompt(
    rng: random.Random,
    available_options: list[str],
) -> tuple[str, str]:
    allowed_options = [
        (suffix, prompt_path)
        for suffix, prompt_path in PROMPT_OPTIONS
        if suffix in available_options
    ]

    if not allowed_options:
        raise RuntimeError("No prompt options are enabled.")

    suffix, prompt_path = rng.choice(allowed_options)
    prompt = load_text_file(prompt_path)
    if not prompt:
        raise RuntimeError(f"Prompt not found or unreadable: {prompt_path}")
    return suffix, prompt


def build_export_path(output_root: Path, provider: str, dataset_name: str, scene_name: str, prompt_suffix: str) -> Path:
    return output_root / sanitize_provider_name(provider) / dataset_name / f"{scene_name}_{prompt_suffix}.png"


def translate_scene_gemini(
    client: genai.Client,
    scene_dir: Path,
    prompt: str,
    export_path: Path,
) -> bool:
    images = collect_scene_inputs(scene_dir)
    if images is None:
        return False

    contents = [prompt, *images]
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=gemini_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=gemini_types.ImageConfig(
                        aspect_ratio="4:3",
                        image_size="1K",
                    ),
                ),
            )

            output_img = extract_first_gemini_image_from_response(response)
            if output_img is None:
                print("Error: Gemini API returned no image output.")
                debug_gemini_response(response)
                return False

            output_img = resize_to_target(output_img)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(export_path)
            return True

        except Exception as exc:
            last_error = exc
            print(f"Warning: Gemini API attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    print(f"Error: Gemini API failed after {MAX_RETRIES} attempts: {last_error}")
    return False


def translate_scene_openai(
    client: OpenAI,
    scene_dir: Path,
    prompt: str,
    export_path: Path,
) -> bool:
    images = collect_scene_inputs(scene_dir)
    if images is None:
        return False

    image_uploads = []
    for image_index, image in enumerate(images):
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_uploads.append(
            (
                f"{scene_dir.name}_{image_index}.png",
                buf.getvalue(),
                "image/png",
            )
        )

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.images.edit(
                model=OPENAI_MODEL_NAME,
                image=image_uploads,
                prompt=prompt,
                size=OPENAI_REQUEST_SIZE,
            )

            image_b64 = result.data[0].b64_json if result.data else None
            if not image_b64:
                print("Error: OpenAI API returned no image output.")
                return False

            output_img = Image.open(BytesIO(base64.b64decode(image_b64))).copy()
            output_img = resize_to_target(output_img)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(export_path)
            return True

        except Exception as exc:
            last_error = exc
            print(f"Warning: OpenAI API attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    print(f"Error: OpenAI API failed after {MAX_RETRIES} attempts: {last_error}")
    return False


def translate_scene_qwen(
    pipe,
    scene_dir: Path,
    prompt: str,
    export_path: Path,
    seed: int,
) -> bool:
    images = collect_scene_inputs(scene_dir)
    if images is None:
        return False

    gen_device = "cuda" if torch and torch.cuda.is_available() and pipe.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with torch.inference_mode():
                result = pipe(
                    image=images,
                    prompt=prompt,
                    negative_prompt=QWEN_NEGATIVE_PROMPT,
                    generator=generator,
                    true_cfg_scale=QWEN_TRUE_CFG_SCALE,
                    guidance_scale=QWEN_GUIDANCE_SCALE,
                    num_inference_steps=QWEN_NUM_INFERENCE_STEPS,
                    num_images_per_prompt=1,
                )

            output_img = resize_to_target(result.images[0])
            export_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(export_path)
            return True

        except Exception as exc:
            last_error = exc
            print(f"Warning: Qwen attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    print(f"Error: Qwen failed after {MAX_RETRIES} attempts: {last_error}")
    return False


def get_scene_dirs(subdir: Path, start_index: int, max_scenes: int) -> list[Path]:
    scenes = sorted(
        d for d in subdir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    )
    filtered_scenes = []
    for scene_dir in scenes:
        try:
            scene_index = int(scene_dir.name.split("_")[-1])
        except ValueError:
            continue

        if scene_index >= start_index:
            filtered_scenes.append(scene_dir)

    return filtered_scenes[:max_scenes]


def build_provider_client(args: argparse.Namespace):
    if args.provider == "gemini-5":
        if genai is None or gemini_types is None:
            raise RuntimeError("google-genai is not installed for --provider gemini.")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        return genai.Client(api_key=api_key)

    if args.provider == "openai-5":
        if OpenAI is None:
            raise RuntimeError("openai is not installed for --provider openai.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        return OpenAI(api_key=api_key)

    if args.provider == "qwen-5":
        device = choose_qwen_device(args.qwen_device)
        dtype = choose_qwen_dtype(device)
        print(f"Loading Qwen pipeline: {QWEN_MODEL_ID}")
        print(f"Qwen device: {device} | dtype: {dtype}")
        return load_qwen_pipeline(device=device, dtype=dtype, offload=args.qwen_offload)

    raise RuntimeError(f"Unsupported provider: {args.provider}")


def main() -> int:
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: dataset directory not found: {args.dataset_dir}")
        return 1

    rng = random.Random(args.seed)

    client_or_pipe = None
    if not args.prepare_only:
        try:
            client_or_pipe = build_provider_client(args)
        except Exception as exc:
            print(f"Error: failed to initialize provider '{args.provider}': {exc}")
            return 1

    subdirs = sorted(d for d in args.dataset_dir.iterdir()
                     if d.is_dir() and (args.powder is None or d.name == args.powder))
    print(f"Found {len(subdirs)} subdirectories in Dataset.")
    print(f"Using provider: {args.provider}")

    total_processed = 0
    total_success = 0
    total_skipped = 0
    total_failed = 0

    for subdir in subdirs:
        print(f"\n--- Processing Subdirectory: {subdir.name} ---")
        scenes = get_scene_dirs(subdir, args.start_index, args.max_scenes)
        print(f"Processing {len(scenes)} scenes...")

        for scene_index, scene_dir in enumerate(scenes):
            try:
                prompt_suffix, prompt = choose_prompt(rng, args.available_options)
            except Exception as exc:
                print(f"Error: failed to load prompt for {scene_dir.name}: {exc}")
                total_failed += 1
                continue

            if args.prepare_only:
                total_processed += 1
                try:
                    report_path = prepare_scene_inputs(scene_dir, args.output_root, args.provider,
                                                       subdir.name, prompt_suffix, prompt)
                    print(f"Prepared {scene_dir.name}: {report_path}")
                    total_success += 1
                except Exception as exc:
                    print(f"Preparation failed for {scene_dir.name}: {exc}")
                    total_failed += 1
                continue

            export_path = build_export_path(
                output_root=args.output_root,
                provider=args.provider,
                dataset_name=subdir.name,
                scene_name=scene_dir.name,
                prompt_suffix=prompt_suffix,
            )

            if export_path.exists():
                print(f"Skipping {scene_dir.name} ({prompt_suffix}, already exists)")
                total_skipped += 1
                continue

            print(f"Translating {scene_dir.name} with {prompt_suffix} prompt...")
            total_processed += 1

            if args.provider == "gemini-5":
                success = translate_scene_gemini(client_or_pipe, scene_dir, prompt, export_path)
            elif args.provider == "openai-5":
                success = translate_scene_openai(client_or_pipe, scene_dir, prompt, export_path)
            else:
                qwen_seed = (args.seed if args.seed is not None else QWEN_SEED) + scene_index
                success = translate_scene_qwen(client_or_pipe, scene_dir, prompt, export_path, qwen_seed)

            if success:
                model = {"gemini-5": GEMINI_MODEL_NAME, "openai-5": OPENAI_MODEL_NAME,
                         "qwen-5": QWEN_MODEL_ID}[args.provider]
                provenance = {
                    "model_id": model, "provider_configuration": args.provider,
                    "prompt_file": f"{ {'LC': 'Low', 'MC': 'Medium', 'HC': 'High'}[prompt_suffix] }Contrast5Image.txt",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_order": ["snapshot.png", "instance.png", "normals.png", "ao.png", "depth.png"],
                    "source_scene": f"{subdir.name}/{scene_dir.name}",
                    "seed": (args.seed + scene_index) if args.provider == "qwen-5" else args.seed,
                    "output_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
                }
                export_path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
                print(f"Saved to {export_path}")
                total_success += 1
            else:
                print(f"Failed {scene_dir.name}")
                total_failed += 1

    print("\nSummary")
    print(f"Processed: {total_processed}")
    print(f"Successful: {total_success}")
    print(f"Skipped: {total_skipped}")
    print(f"Failed: {total_failed}")

    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
