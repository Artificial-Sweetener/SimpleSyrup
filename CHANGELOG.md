# [1.6.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.5.0...v1.6.0) (2026-08-09)


### Bug Fixes

* **cache:** make integer narrowing checker-independent ([6e2d1e1](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/6e2d1e1844b361f9b2f3a31c8538cfde0ce7c6b1))
* **mask:** preserve missing-alpha image geometry ([70ffeb5](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/70ffeb530acb1f5e55ac6e06adcc07ba4560d776))
* **media:** stabilize native ordered preview controls ([3d03b1d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/3d03b1dfbc2eef3cee900173ddda57e03d5bc43d))
* **regional:** align prompt batches and LoRA hooks ([646e4e7](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/646e4e7cab0ed299e704c4acbf64849181180076))
* **runtime:** centralize Comfy patcher lifecycle ([fda2ef4](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/fda2ef4074e8550a406ada317f0a3bfb7db39cf6))


### Features

* **conditioning:** add regional prompting and SEP-local LoRAs ([be26493](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/be264935a8cb0e805222de9d615a197a788d8f01))
* **conditioning:** support labeled prompt separators ([a24da13](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/a24da1359cc9966762d8e1fa4fde3dbdef879cfa))
* **loaders:** add automatic FLUX model loaders ([08fd18c](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/08fd18cc6fdf6a7f9ff3657c323b31e8960a235a))
* **media:** add native ordered loaders and SEGS preview ([dbde2a9](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/dbde2a9266dd043b92126e42e502cc88d9fba777))
* **sampling:** add contextual diffusion sampler ([6c2e2da](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/6c2e2dad9c6db42a2a2a82d1e4a17e4db4bb0fbf))
* **sampling:** expose evaluated context SEGS ([53a2771](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/53a2771ce815a0705b92f4766e56f11adbd9dedb))
* **segmentation:** add interactive SEGS preview ([21db62d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/21db62d3acee0b1d087f83e01fb5b92c912f4df6))
* **segmentation:** add SAM region overlay ([d1ead71](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/d1ead71ca714e365bf27448170e6c8c8fccd2f5a))
* **segmentation:** add SAM-guided tiled diffusion ([e50ec0e](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/e50ec0ea79620af63f32c25a934b2b06fc4d9484))

# [1.5.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.4.0...v1.5.0) (2026-07-14)


### Bug Fixes

* **groundingdino:** support transformers v4 and v5 ([239070b](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/239070b12d3eecbfd5c47c9410c7ca31ac1402ac))


### Features

* **masking:** expand segmentation tooling and progress ([f70766d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/f70766dafe096895ad8d6309681fd59270664600))

# [1.4.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.3.0...v1.4.0) (2026-06-02)


### Bug Fixes

* **tiled-diffusion:** clamp overlap for small latents ([d7448c6](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/d7448c6ca52ce517b8d0f8ee697249c5def13535))


### Features

* **detailing:** add external llm segs tagging ([b7cd40c](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/b7cd40c85ae9f9514d0b1ff3c3f6007730796752))
* **segs:** add regional batching and wd14 tagging nodes ([fef60ad](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/fef60adeca2c8ec7c0641e8106bb0863ee2f195e))

# [1.3.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.2.0...v1.3.0) (2026-05-26)


### Features

* **prompt-control:** add schedule and encode prompt node ([bd515e6](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/bd515e696cedcc78d056af6c23b9193e34f131bc))

# [1.2.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.1.0...v1.2.0) (2026-05-25)


### Features

* **nodes:** add VAE options and clone-safe diffusion ([6ff2dc8](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/6ff2dc8c24a6f7ddde3182b81bcbe6aad65427f4))

# [1.1.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.0.0...v1.1.0) (2026-05-23)


### Bug Fixes

* **detailers:** align SEGS mask blending behavior ([74c83a6](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/74c83a6a70407aea6a34b0c0e31408f32b929882))
* use SimpleSyrup package identity ([4fa5582](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/4fa5582796e9ace2a8805cb1ea2aebce65551e89))


### Features

* **detection:** add keep-only SEGS selection ([8f8ee91](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/8f8ee91dea3ce1a044c0e61b482e571c51b372bc))

# 1.0.0 (2026-05-22)


### Features

* initial release ([e513baf](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/e513baf70a20306856e40fbf2afd80b25f5655a6))

# Changelog

All notable changes to this project will be documented in this file.
