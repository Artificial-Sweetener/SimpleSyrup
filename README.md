# SimpleSyrup

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE) [![Comfy Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2FSimpleSyrup&query=%24.latest_version.version&label=Comfy%20Registry&color=5b5bd6)](https://registry.comfy.org/publishers/artificialsweetener/nodes/SimpleSyrup) [![Comfy Registry downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2FSimpleSyrup&query=%24.downloads&label=downloads&color=5b5bd6)](https://registry.comfy.org/publishers/artificialsweetener/nodes/SimpleSyrup) [![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**SimpleSyrup** is a ComfyUI node pack that grew out of moving my A1111/WebUI image workflows into ComfyUI graphs.

The WebUI side shows up in the things I kept reaching for: ADetailer-style inline `[SEP]` prompt batches, tiled diffusion, familiar checkpoint loader controls, CLIP skip where WebUI users expect it, and sampler/scheduler extras. The Comfy side matters just as much: the detailers are modeled heavily on ComfyUI Impact Pack's SEGS workflow, and the utility nodes are built for graph readability, ordered data, and explicit runtime behavior.

SimpleSyrup pulls from a few different places:

- Moving from **A1111/WebUI** to **ComfyUI** is the reason this pack brings tiled diffusion, familiar checkpoint loader control grouping, ADetailer-style prompt splitting, and A1111-flavored sampler behavior into node graphs.
- **ComfyUI Impact Pack** is the main influence for the SEGS and detailer shape: detectors create SEGS, detailers sample cropped areas, masks are feathered, and results are composited back into the source image.
- **ADetailer** is the influence for inline `[SEP]` per-segment prompt batches.
- The remaining utility pieces cover practical graph needs: GPU Lanczos resizing through TorchLanc, latent provenance helpers, and smaller nodes for image, prompt, and conditioning workflows.

## Highlights

- Impact-compatible SEGS detection, sorting, combining, tiling, and detailing.
- Detailer nodes modeled heavily on ComfyUI Impact Pack's SEGS workflow.
- ADetailer-style inline `[SEP]` per-segment prompt batches.
- MultiDiffusion, Mixture of Diffusers, and regional MultiDiffusion sampling paths.
- A WebUI-familiar checkpoint loader with CLIP skip and VAE override controls.
- A Simple Anima loader that keeps Anima's model, VAE, dtype, and device controls together.
- Loaders for SAM, GroundingDINO, ViTMatte, Ultralytics, and WD14.
- Tile & Tag SEGS workflows that run WD14 on deterministic tile crops and keep conditioning aligned to tile order.
- KSampler extras including A1111-style Euler ancestral behavior, AYS, GITS, automatic A1111 scheduling, and beta57.
- GPU Lanczos resizing through [TorchLanc](https://github.com/Artificial-Sweetener/TorchLanc), with batch and mask handling.
- Provenance-aware latent helpers for recovering the latent behind an unmodified decoded image.
- Settings-backed model dropdowns that can show known downloadable models or only locally installed ones.

## Installation

**Recommended: install through ComfyUI Manager**

Open **Manager** from the ComfyUI toolbar, click **Custom Nodes Manager**, search for **SimpleSyrup**, and click **Install**. Restart ComfyUI after installation.

**Manual install**

If you would rather install it yourself, clone this repo into `ComfyUI/custom_nodes/`, activate your **ComfyUI venv**, and install this node pack's requirements.

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/Artificial-Sweetener/SimpleSyrup.git
cd SimpleSyrup
pip install -r requirements.txt
```

ComfyUI already provides the heavy shared runtime stack, including PyTorch. SimpleSyrup adds the packages it needs for specific features, including TorchLanc, Ultralytics, ONNX Runtime, Segment Anything, and Hugging Face download helpers.

## The Nodes

SimpleSyrup is organized around workflow jobs, not socket types.

### Impact-Style SEGS Workflows

SimpleSyrup speaks the Impact Pack SEGS shape on purpose. It can read Impact-style SEGS, sort them, combine them, tile them, and emit SEGS payloads that Impact-style consumers can read.

The detailer nodes are modeled heavily on **ComfyUI Impact Pack**. Detectors create SEGS, SEGS choose the crop areas, crops are sampled, masks are feathered, and the results are composited back into the source image.

- **Prompt SEGS w/ SAM** uses GroundingDINO to find prompt-matched boxes, SAM to segment them, optional negative prompting to subtract unwanted areas, and optional ViTMatte refinement to clean up mask edges. It returns both SEGS and a combined mask.
- **Detect SEGS w/ Ultralytics** runs bbox or segmentation detection, filters by threshold and size, supports label filtering, and returns Impact-compatible SEGS plus a combined mask.
- **Detail SEGS by Scale Factor** upscales each SEG crop, samples it, downsizes it back, and composites it into the original image with feathering and optional denoise masks.
- **Detail SEGS by Scale Factor w/ Tiled Diffusion** uses the same crop/detail idea, but samples large crops through SimpleSyrup's tiled diffusion path.
- **Detail SEGS as Regions** runs one regional MultiDiffusion pass over the image and pairs every SEG with its matching `CONDITIONING_BATCH` entry.

The regional node is different from the per-crop detailers. It samples one full-image latent with regional conditioning, using the global prompt for full-image context while each SEG gets its own positive conditioning.

### Per-Segment Prompt Batches

SimpleSyrup layers the ADetailer habit I missed from WebUI on top of the Impact-style detailer shape: writing per-segment prompt batches inline with `[SEP]`.

Those prompts become an ordered `CONDITIONING_BATCH`, so prompt 1 stays matched to SEG 1, prompt 2 stays matched to SEG 2, and so on. This keeps the graph readable when each detected item needs its own prompt.

- **Encode Prompt Batch** splits prompt text with `[SEP]` and encodes ordered positive and negative `CONDITIONING_BATCH` values.
- **Conditioning Batch Start** and **Conditioning Batch Append** build ordered conditioning batches for per-segment and regional workflows.

### Tile, Tag, and Guide

**Tile & Tag SEGS** is for workflows where tile regions should carry their own generated prompt guidance.

It splits an image into deterministic tile SEGS, crops each tile, runs WD14 tagging on each crop, prefixes your universal positive prompt text, and CLIP-encodes the resulting prompts into a `CONDITIONING_BATCH`. The order matters: the conditioning batch is aligned to the tile SEGS order so downstream per-SEG or regional nodes can pick the right prompt for the right area.

That is the kind of thing that is easy to do once by hand and annoying to keep correct in a real graph.

### Tiled Sampling

**KSampler (Tiled Diffusion)** is a KSampler-style node with selectable **MultiDiffusion** and **Mixture of Diffusers** modes.

It splits the latent into tiles, denoises tile predictions, and blends them back together during sampling. MultiDiffusion averages overlapping predictions. Mixture of Diffusers uses weighted blending. Both are there because large images and large upscale passes often need a different strategy than normal full-latent denoising.

Tiled diffusion is here because it was one of the high-resolution workflow tools I kept reaching for in my WebUI setup. SimpleSyrup brings MultiDiffusion and Mixture of Diffusers behavior into normal Comfy sampling nodes, so large latent jobs can be tiled without giving up Comfy's explicit conditioning and graph wiring.

This also matters for Anima workflows. Anima can produce beautiful images, but pushing beyond its comfortable native size with untiled diffusion upscale can smear detail instead of improving it. The tiled path gives those workflows another route.

### Diffusion Loaders

**Simple Load Checkpoint** is meant to feel familiar if you come from WebUI, where the common generation controls live near the model selection.

- **Simple Load Checkpoint** loads a checkpoint, optionally replaces the checkpoint VAE, and keeps CLIP skip in the same place.
- **Simple Load Anima** is for Anima workflows. It loads Anima with the Qwen text encoder and Qwen image VAE it expects. You can choose the files yourself or let SimpleSyrup resolve the known Anima assets automatically. It keeps model, VAE, dtype, and device decisions together so the rest of the graph can get on with the image.

### Model and Detector Loaders

These nodes load the models used by detection, segmentation, tagging, matting, and compatibility workflows.

- **SAM Model Loader** loads SAM, SAM-HQ, and MobileSAM choices for segmentation workflows.
- **GroundingDINO Model Loader** loads GroundingDINO with an explicit BERT text encoder.
- **ViTMatte Model Loader** loads ViTMatte for mask edge refinement.
- **Load Ultralytics Model** loads an Ultralytics detector and exposes both SimpleSyrup's native detector model and Impact-style compatibility outputs.
- **Load WD14 Tagger** loads a SmilingWolf WD14 ONNX model and its tag CSV.
- **LayerStyle SAM Models Adapter** splits a LayerStyle `LS_SAM_MODELS` bundle into separate `SAM_MODEL` and `DINO_MODEL` outputs.
- **Grounded SAM Model Info** returns JSON metadata for selected SAM and GroundingDINO models.

The LayerStyle adapter exists because good ComfyUI workflows should not make you reload the same SAM or GroundingDINO model just because one node pack uses a different socket shape.

### Sampler and Scheduler Extras

**KSampler (Extras)** keeps the normal Comfy sampler shape, but adds sampler and scheduler behavior I wanted available without dragging in a separate sampler stack.

It includes:

- `euler_a_a1111`, an A1111/k-diffusion-style Euler ancestral sampler.
- **AYS SD1** and **AYS SDXL** schedules.
- **GITS**.
- **automatic_a1111** scheduler behavior.
- **beta57**, a local reimplementation of the RES4LYF beta57 scheduler preset.

The node still uses Comfy-style seed handling, partial denoise behavior, progress callbacks, and normal positive/negative conditioning inputs.

### Image, Prompt, and Latent Utilities

These nodes handle the smaller jobs that show up all over image workflows.

- **Resize Image to Target** resizes image batches with stretch, keep-aspect, crop, and pad modes. It can round output dimensions to a divisibility target, anchor crop or pad placement, process batches in chunks, resize a mask with the image, and use GPU Lanczos through [TorchLanc](https://github.com/Artificial-Sweetener/TorchLanc).
- **Simple VAE Encode** encodes an image to latent space, but reuses the source latent when the graph proves the image came from an unmodified `VAEDecode`.
- **Upscale Latent From Image** finds the latent behind an unmodified decoded image and expands to Comfy's latent upscale behavior.
- **Latent Diagnostics** passes a latent through unchanged while reporting shape, dtype, device, and tiling-fit details.
- **Prompt Encode Style** creates Prompt Control style tags from an encode-style selection.
- **Prompt Encode Style & Normalization** creates Prompt Control style and normalization tags together.
- **Scale Factor** provides a bounded scale multiplier for nodes that expect one.
- **Seed** provides a reusable seed value with ComfyUI seed controls.

The provenance nodes trace the graph. They do not guess from tensor values. If an image has been loaded, edited, cropped, detailed, resized, or otherwise changed, the original latent provenance is broken and the node will not pretend otherwise.

## Settings

SimpleSyrup adds one ComfyUI setting:

- **SimpleSyrup: Show downloadable models in loader dropdowns**

When this is enabled, supported loaders show known downloadable model choices even if the files are not installed yet. When it is disabled, those dropdowns only show models SimpleSyrup can verify locally.

This setting affects SAM, GroundingDINO, ViTMatte, and WD14 loader dropdowns. Anima's automatic Qwen text encoder and VAE resolution is handled by the Anima loader itself.

## License & Acknowledgements

**SimpleSyrup** is licensed under the GNU Affero General Public License v3.0 or later (**AGPL-3.0-or-later**). Please read the full [LICENSE](LICENSE) included with this repo.

AGPL-3.0-or-later is a strong copyleft license. If you convey SimpleSyrup or a modified version, you must provide the corresponding source; and if you let users interact with a modified version over a network, you must offer those users the corresponding source for that modified version.

SimpleSyrup owes a lot to other projects:

- [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) for the SEGS workflow vocabulary and detailer shape this pack is heavily modeled around.
- [ADetailer](https://github.com/Bing-su/adetailer) for the inline `[SEP]` per-segment prompt workflow I missed from WebUI.
- [ComfyUI Layer Style Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance) for the SAM workflow surface this pack interoperates with.
- [Tiled Diffusion & VAE for AUTOMATIC1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) for practical tiled diffusion and Mixture of Diffusers behavior.
- [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) for the beta57 scheduler preset reimplemented here.

SimpleSyrup also vendors or reimplements selected third-party behavior for SAM-HQ, MobileSAM, GroundingDINO, AUTOMATIC1111 sampler behavior, k-diffusion, and tiled diffusion behavior. See [third_party/NOTICE.md](third_party/NOTICE.md) for the full third-party notices.

### Research Citations

SimpleSyrup's tiled diffusion behavior is based on ideas from MultiDiffusion and Mixture of Diffusers.

```bibtex
@article{bar2023multidiffusion,
  title={MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation},
  author={Bar-Tal, Omer and Yariv, Lior and Lipman, Yaron and Dekel, Tali},
  journal={arXiv preprint arXiv:2302.08113},
  year={2023}
}
```

```bibtex
@article{barbero2023mixture,
  title={Mixture of Diffusers for scene composition and high resolution image generation},
  author={Barbero Jimenez, Alvaro},
  journal={arXiv preprint arXiv:2302.02412},
  year={2023}
}
```

## From the Developer 💖


- **Buy Me a Coffee**: You can help fuel more projects like this at my [Ko-fi page](https://ko-fi.com/artificial_sweetener).
- **My Website & Socials**: See my art, poetry, and other dev updates at [artificialsweetener.ai](https://artificialsweetener.ai).
- **If you like this project**, it would mean a lot to me if you gave me a star here on GitHub!! ⭐
