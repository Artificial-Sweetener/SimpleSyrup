# Third-Party Notices

This repository vendors selected third-party behavior for runtime use. Each
vendored component is recorded in `third_party/manifest.toml`, and the
corresponding license text is stored in `third_party/licenses/`.

## SAM-HQ and MobileSAM runtime

SimpleSyrup vendors selected SAM-HQ and MobileSAM runtime files under
Apache-2.0 for loading SAM-family segmentation models inside ComfyUI. The
vendored runtime is kept under `simple_syrup/third_party/sam_hq_runtime/`
and preserves upstream copyright notices where the source files carried them.

## GroundingDINO runtime

SimpleSyrup vendors selected GroundingDINO runtime files under Apache-2.0 for
prompt-based box detection. The vendored runtime is kept under
`simple_syrup/third_party/groundingdino_runtime/` and preserves upstream
copyright notices where the source files carried them.

## RES4LYF beta57 scheduler preset

SimpleSyrup vendors the RES4LYF `beta57` scheduler preset under AGPL-3.0.
The preset uses ComfyUI's beta scheduler with `alpha=0.5` and `beta=0.7`.
SimpleSyrup resolves the preset locally for `KSampler (Extras)` and does not
patch ComfyUI's global scheduler registry.

## AUTOMATIC1111 Euler a sampler integration

SimpleSyrup vendors selected AUTOMATIC1111 WebUI sampler integration behavior
under AGPL-3.0. This provenance covers the `Euler a` sampler mapping, the
`Automatic` scheduler fallback behavior, and the documented decision not to
port AUTOMATIC1111 ENSD or RNG hijacking behavior.

## k-diffusion Euler ancestral sampler

SimpleSyrup vendors the Euler ancestral sampler loop from k-diffusion under
the MIT license. The local sampler keeps the A1111/k-diffusion loop structure
while running inside ComfyUI's deterministic seed system.

## Mixture of Diffusers and MultiDiffusion tiled diffusion behavior

SimpleSyrup reimplements tiled denoising behavior after
inspecting the local `multidiffusion-upscaler-for-automatic1111` extension,
which is licensed under CC-BY-NC-SA-4.0. The implementation preserves the
extension's latent tile planning, Mixture Gaussian tile weighting,
MultiDiffusion uniform tile averaging, regional prompt mask blending, and
pre-CFG model prediction blending behavior without importing the extension at
runtime.

## SmilingWolf WD tagger models

SimpleSyrup's `Tile & Tag SEGS` node can download selected SmilingWolf WD
tagger ONNX models and `selected_tags.csv` files at runtime from Hugging Face.
These model files are not vendored in this repository. The runtime catalog
points to the corresponding `SmilingWolf/*` repositories and stores downloaded
files in the user's ComfyUI model directory.
