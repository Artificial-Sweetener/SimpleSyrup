# SimpleSyrup

[![Comfy Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2FSimpleSyrup&query=%24.latest_version.version&label=Comfy%20Registry&color=5b5bd6)](https://registry.comfy.org/publishers/artificialsweetener/nodes/SimpleSyrup) [![Comfy Registry downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.comfy.org%2Fnodes%2FSimpleSyrup&query=%24.downloads&label=downloads&color=5b5bd6)](https://registry.comfy.org/publishers/artificialsweetener/nodes/SimpleSyrup) [![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/) [![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)

**SimpleSyrup** is a ComfyUI node pack that grew out of moving my A1111/WebUI image workflows into ComfyUI.

The WebUI influence shows up all over the pack. I missed ADetailer's inline prompt batches, tiled diffusion, CLIP skip beside my checkpoint, and some of the sampler behavior I was used to. I also wanted the regional pieces to work with Impact Pack SEGS so I could use the same regions across detectors and detailers.

The pack now covers model loading, regional prompting and segmentation, high-resolution sampling, image and mask utilities, tagging, and the smaller pieces I need to keep those workflows readable.

[**SugarSubstitute**](https://github.com/Artificial-Sweetener/SugarSubstitute), my native desktop front-end for ComfyUI, can use Cubes built from any ComfyUI nodes available in its connected environment. Its first-party [**Base-Cubes**](https://github.com/Artificial-Sweetener/Base-Cubes) pack uses SimpleSyrup for model loading, regional prompts, segmentation, high-resolution sampling, and other graph work. You can also install SimpleSyrup on its own and use the nodes in normal ComfyUI workflows.

## Highlights

- Loaders that keep checkpoints, Anima, FLUX.1, and FLUX.2 models together with the text encoders, VAE, precision, and device choices they need.
- My original Contextual Diffusion method for coherent high-resolution edits, plus MultiDiffusion and Mixture of Diffusers tiled sampling.
- Impact-compatible SEGS detection, segmentation, interactive preview, batching, and detailers.
- ADetailer-style `[SEP]` prompt batches, masked conditioning, and regional samplers, with optional Prompt Control scheduling and LoRA hooks.
- WD14 and external vision LLM tagging that stays aligned with the right regions.
- Ordered image and mask loading, GPU Lanczos resizing, tiled VAE options, and provenance-aware latent tools.
- WebUI-inspired sampling extras including seed variation, A1111 Euler ancestral behavior, AYS, GITS, `automatic_a1111`, and beta57.

## Contents

- [Install](#install)
- [Model loading](#model-loading)
- [Large images and high-resolution edits](#large-images-and-high-resolution-edits)
  - [Contextual Diffusion](#contextual-diffusion)
  - [Tiled Diffusion](#tiled-diffusion)
- [SEGS, detailers, and regional prompts](#segs-detailers-and-regional-prompts)
- [Tagging images and regions](#tagging-images-and-regions)
- [Images, masks, latents, and sampler extras](#images-masks-latents-and-sampler-extras)
- [Settings and optional integrations](#settings-and-optional-integrations)
- [License, acknowledgements, and research](#license-acknowledgements-and-research)

## Install

### ComfyUI Manager

Open Manager and search the **Node Pack** list for **SimpleSyrup**, then select it and click **Install**. Restart ComfyUI when it finishes.

ComfyUI still has two Manager interfaces in circulation. In the legacy interface, the search is under **Custom Nodes Manager**.

### Manual install

Clone the repository into `ComfyUI/custom_nodes/` and install the requirements with the same Python environment that runs ComfyUI.

For a normal Windows virtual environment:

```powershell
Set-Location ComfyUI\custom_nodes
git clone https://github.com/Artificial-Sweetener/SimpleSyrup.git
Set-Location SimpleSyrup
..\..\venv\Scripts\python.exe -m pip install -r requirements.txt
```

For ComfyUI Windows Portable, run this from the portable installation folder after cloning the repository:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\SimpleSyrup\requirements.txt
```

Restart ComfyUI after installation. SimpleSyrup uses ComfyUI's v3 extension API, so you need a current version of ComfyUI for the nodes to appear.

ComfyUI already supplies PyTorch and the rest of the shared runtime. SimpleSyrup installs the packages used by its own features, including [TorchLanc](https://github.com/Artificial-Sweetener/TorchLanc), Ultralytics, ONNX Runtime, Segment Anything, and the Hugging Face download helpers.

## Model loading

Loading a checkpoint used to feel like choosing one file. Newer model families can mean a diffusion model, several text encoders, a VAE, and then the precision and device choices for all of them. I made the SimpleSyrup loaders so I could deal with that setup once and get on with the workflow.

**Simple Load Checkpoint** is the normal checkpoint loader. It has an optional VAE override and keeps CLIP skip beside the model controls where I expect to find it.

**Simple Load Anima** loads Anima with its Qwen text encoder and Qwen image VAE. You can select every part yourself. If you don't want to, the automatic choices can find and download the known checksum-pinned support files.

**Simple Load FLUX** handles FLUX.1 with CLIP-L, T5-XXL, and its VAE. **Simple Load FLUX.2** inspects the selected diffusion model and chooses the matching text encoder family for FLUX.2 dev, Klein 4B, or Klein 9B/KV conditioning. Both loaders can find or download their known text encoders and VAEs with visible Comfy progress.

The FLUX loaders only download those revision-locked, checksum-pinned support files. You still install and select the diffusion model. They also expose manual component selection, diffusion weight precision, and text-encoder device placement. Moving text encoding to the CPU can save VRAM, although it will take longer.

## Large images and high-resolution edits

Tiled diffusion handles the obvious large-image problem: sometimes the latent is too big to evaluate all at once. There is a worse version. The model has enough memory to run, but the canvas is so far outside its normal working resolution that it starts making terrible decisions anyway.

Tiling keeps each evaluation small. It does not make the tiles understand the same complete image. That second problem is why I made Contextual Diffusion.

### Contextual Diffusion

**KSampler (Contextual Diffusion)** is an original sampling method I developed for editing and refining oversized latent canvases.

The first real target was a 2160 × 3072 source image I wanted to edit with FLUX.2 Klein 4B. Downscaling made the edit coherent, but that defeated the point of starting with a high-resolution source. Ordinary tiled diffusion kept much more detail and looked promising at first. Then I looked at the whole image. One tile had found a figure, another had invented a second figure, and different parts of the cathedral had become different buildings. The overlaps were smooth! The scene was still nonsense.

I needed Klein to see the complete composition and the full-resolution detail during the same denoising process. Contextual Diffusion does that by making overlapping local predictions on the original latent and a second prediction from a smaller, aspect-preserving view of the whole image during the early steps.

That took some trial and error. Directly blending the whole-image prediction into the tiles made the result blurry. Leaving it active too long produced smears, repeated edges, and other low-resolution garbage in the final detail. What finally worked was subtracting the low-frequency interpretation already present in the tiled prediction and adding only the difference from the whole-image prediction:

`prediction = local + scheduled_weight × (global_upsampled − local_low_frequency)`

The whole-image correction is strongest at the beginning and can decay before the model starts settling fine texture. Distilled Klein models commonly finish in four steps, so even one corrected step is already a quarter of the denoising process.

Contextual Diffusion is for edits the model already knows how to make at a normal resolution. I use it for clothing, material, color, jewelry, expression, local lighting, and other changes where I want to keep the source pose and composition. If I need a completely new pose, camera, and environment, I establish those at a normal working resolution first and refine the result afterward.

FLUX.2 reference latents stay complete and ordered in every local and whole-image evaluation. This lets one image retain the target composition while other images continue to provide complete subject or style references. The sampler also supports Anima's singleton-depth latent shape, which is useful when refining an illustration after a conventional resize.

You can connect SEGS to replace the normal grid with a region-guided context plan. This gives you some control over where the local windows fall. Earlier versions ran regular tiles and a second bank of SAM views at the same time because I thought more views of the important objects would help. Instead, I got duplicated hats, extra limbs, repeated garment edges, and other semantic echoes. It was the wrong architecture, so I removed it. The current method uses one local plan at a time and returns its actual windows through `contexts_segs` so you can see what it evaluated.

The cost is one tiled prediction pass per denoising step and one smaller whole-image evaluation for each step using the correction. UniPC, regional conditioning, ControlNet, and GLIGEN are currently rejected because I haven't validated their spatial behavior across both context sizes.

I found the formula by comparing the failures and adjusting the method until the whole-image branch could fix composition without taking the detail away from the tiles. After I had implemented it, I learned about [Upsample Guidance](https://arxiv.org/abs/2404.01709). It uses a closely related separation between low-frequency guidance and a high-resolution residual.

Upsample Guidance wasn't part of how I developed Contextual Diffusion. There are also practical differences: my high-resolution prediction is assembled from bounded tiles, the complete image is fit into an aspect-preserving context, reference latents stay whole, and the node has its own early-step controls and optional SEGS planning. Still, the mathematical relationship is real. My experiments are qualitative, and I describe the method as independent development of a related multiscale idea instead of claiming priority over that paper.

### Tiled Diffusion

**KSampler (Tiled Diffusion)** is the more direct tiled sampler. It divides the latent into overlapping contexts, evaluates them in batches, and combines the predictions during every denoising step.

MultiDiffusion averages the overlapping predictions. Mixture of Diffusers uses Gaussian weights that favor the center of each context. This works well when local evaluation and overlap blending are enough for the image. Contextual Diffusion adds the whole-image correction for edits where the separate contexts lose track of the complete scene.

The same tiled sampling path is available in **Detail SEGS by Scale Factor w/ Tiled Diffusion** for large detailer crops and **KSampler (Prompt by Tiled Region)** for regional prompts on large canvases.

## SEGS, detailers, and regional prompts

Impact Pack already had a useful way to represent detected and masked regions: `SEGS`. I built SimpleSyrup around the same shape so regions can move between compatible detectors, these nodes, and Impact workflows without reloading models or rebuilding the masks.

**Prompt SEGS w/ SAM** uses GroundingDINO to find objects from text and SAM to segment them. It also supports negative prompting and optional ViTMatte edge refinement. **Detect SEGS w/ Ultralytics** creates regions from bounding-box or segmentation models with confidence, label, and size filtering. Existing masks can enter the same workflow through **Mask to SEGS**, while **SEGS from SAM Output** runs automatic unprompted segmentation from a connected SAM model.

**Simple Preview SEGS** shows the regions over the image, lets you select them from an interactive grid, and passes the original SEGS onward. **Batch SEGS** combines several ordered SEGS inputs.

The scale-factor detailers work on one crop at a time. They enlarge the crop, sample it, shrink it back, and composite it into the source image with feathering and optional denoise masks. **Detail SEGS as Regions** takes another route: it keeps the full image in one MultiDiffusion pass, uses the global conditioning across the image, and pairs each SEG with its own ordered regional conditioning.

The prompt batching came directly from ADetailer. **Encode Prompt Batch** splits positive and negative text with `[SEP]`. You can write `[SEP|name]` to keep a long prompt readable; matching still follows the order of the prompts and regions. The first prompt is global, and each later prompt belongs to the corresponding mask or SEG.

**Conditioning Batch Start**, **Conditioning Batch Append**, and **Batch Region Conditioning** build the same ordered structure from existing conditioning. **Compose Regional Conditioning** converts a global-first prompt batch and ordered masks into normal masked Comfy conditioning. The dedicated **KSampler (Prompt by Region)** and tiled version apply the regional prompt batch during sampling.

If [ComfyUI Prompt Control](https://github.com/asagi4/comfyui-prompt-control) is installed, SimpleSyrup also exports **Encode Prompt Batch w/ Prompt Control** and **Schedule & Encode Prompts**. They preserve Prompt Control scheduling and LoRA hooks across `[SEP]` regions. The rest of the pack loads normally when Prompt Control is absent.

Model loading stays separate from detection. There are loaders for SAM, GroundingDINO, ViTMatte, and Ultralytics. **LayerStyle SAM Models Adapter** accepts a ComfyUI Layer Style Advance `LS_SAM_MODELS` bundle and exposes the loaded SAM and GroundingDINO models through the normal sockets used here.

## Tagging images and regions

**Load WD14 Tagger** loads a SmilingWolf WD14 ONNX model and its tag CSV. **Tag SEGS w/ WD14** runs the tagger on existing SEG crops and keeps the resulting conditioning in the same order. **Tile & Tag SEGS** makes a deterministic set of tile regions, tags each crop, prefixes shared positive text, and returns the SEGS together with their matching conditioning batch.

The external LLM nodes use a configured OpenAI-compatible provider. **Tag SEGS w/ External LLM** sends each region crop to a vision-capable model and returns aligned conditioning. **External LLM Prompt** sends system and user prompts and returns the response as text, with an optional image for models that support vision.

## Images, masks, latents, and sampler extras

**Load Image List** loads files in selection order as separate image list items, so each image keeps its own dimensions. **Load Mask Batch** loads same-sized files as one `BHW` mask batch and applies the selected channel consistently to every file.

**Resize Image to Target** handles stretch, keep-aspect, crop, and pad modes. It supports divisibility rounding, anchored crop and pad placement, chunked batches, paired masks, and GPU Lanczos through TorchLanc.

**VAE Encode (Options)** and **VAE Decode (Options)** put the normal and tiled VAE paths behind one explicit tiling control, including spatial and temporal tile settings where Comfy supports them.

**Simple VAE Encode** can reuse the source latent when the graph proves that its image came directly from an unmodified `VAEDecode`. **Upscale Latent From Image** uses the same provenance to find and resize the original latent. Loading, editing, cropping, detailing, or resizing the image breaks that provenance. These nodes follow the graph instead of trying to identify a latent from the finished tensor.

**KSampler (Extras)** adds the A1111/k-diffusion-style `euler_a_a1111` sampler, AYS SD1 and SDXL schedules, GITS, the `automatic_a1111` scheduler, and a local implementation of the RES4LYF beta57 preset. It keeps Comfy's regular seed handling, partial denoise behavior, progress callbacks, and conditioning inputs.

**Seed Variation** patches a MODEL so Comfy-native samplers mix their normal initial noise toward a second deterministic seed. Strength `0` keeps the sampler seed unchanged, while strength `1` uses variation-seed initial noise. Ancestral and SDE samplers continue to use the sampler seed for additional noise introduced after initialization.

The remaining utilities are **Latent Diagnostics**, **Scale Factor**, and **Seed**. Latent Diagnostics reports the latent shape, dtype, device, and tiled-sampling compatibility while passing it through unchanged.

## Settings and optional integrations

SimpleSyrup adds three ComfyUI settings:

- **SimpleSyrup: Show downloadable models in loader dropdowns** controls whether known downloadable SAM, GroundingDINO, ViTMatte, and WD14 choices appear before they are installed.
- **SimpleSyrup: External LLM endpoint** stores the OpenAI-compatible base URL used to discover provider models and run the external prompt nodes.
- **SimpleSyrup: External LLM API key** stores the provider key in OS credential storage.

With downloadable models enabled, selecting a known missing catalog entry lets its loader download the required files. With the setting disabled, the dropdowns contain models SimpleSyrup can verify locally. Anima, FLUX.1, and FLUX.2 support components are resolved by their own loaders and use checksum-pinned automatic choices.

Saving the external LLM endpoint and API key refreshes the provider models available in connected SimpleSyrup nodes. Image inputs require a provider model with vision support.

SimpleSyrup currently interoperates with:

- [ComfyUI Prompt Control](https://github.com/asagi4/comfyui-prompt-control) for scheduled prompts and regional LoRA hooks.
- ComfyUI Impact Pack through compatible `SEGS` values.
- ComfyUI Layer Style Advance through its `LS_SAM_MODELS` bundle.

## License, acknowledgements, and research

**SimpleSyrup** is licensed under the GNU Affero General Public License v3.0 or later (**AGPL-3.0-or-later**). Please read the full [LICENSE](LICENSE) included with this repository.

AGPL-3.0-or-later is a strong copyleft license. If you convey SimpleSyrup or a modified version, you must provide the corresponding source. If users interact with a modified version over a network, you must offer those users the corresponding source for that version.

SimpleSyrup owes a lot to other projects:

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) provides the engine and graph ecosystem this pack runs on.
- [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) established the SEGS workflow vocabulary and detailer structure used here.
- [ADetailer](https://github.com/Bing-su/adetailer) is where the inline `[SEP]` per-segment prompt workflow came from.
- [ComfyUI Prompt Control](https://github.com/asagi4/comfyui-prompt-control) provides the scheduled prompt and LoRA-hook behavior used by the optional integration.
- [ComfyUI Layer Style Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance) provides the SAM model bundle SimpleSyrup can adapt.
- [Tiled Diffusion & VAE for AUTOMATIC1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) informed the practical tiled diffusion and Mixture of Diffusers behavior reimplemented here.
- [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) is the source of the beta57 scheduler preset reimplemented here.

SimpleSyrup also vendors or reimplements selected third-party behavior for SAM-HQ, MobileSAM, GroundingDINO, AUTOMATIC1111 sampler behavior, k-diffusion, and tiled diffusion. See [third_party/NOTICE.md](third_party/NOTICE.md) for the complete notices.

### Research citations

SimpleSyrup's tiled diffusion behavior builds on MultiDiffusion and Mixture of Diffusers. Contextual Diffusion was developed independently and was later found to share a related multiscale residual principle with Upsample Guidance.

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

```bibtex
@article{hwang2024upsample,
  title={Upsample Guidance: Scale Up Diffusion Models without Training},
  author={Hwang, Juno and Park, Yong-Hyun and Jo, Youngjung},
  journal={arXiv preprint arXiv:2404.01709},
  year={2024}
}
```

## From the Developer 💖

- **Buy Me a Coffee**: You can help fuel more projects like this at my [Ko-fi page](https://ko-fi.com/artificial_sweetener).
- **My Website & Socials**: See my art, poetry, research notes, and other development updates at [artificialsweetener.ai](https://artificialsweetener.ai).
- **If you like this project**, it would mean a lot to me if you gave me a star here on GitHub!! ⭐
