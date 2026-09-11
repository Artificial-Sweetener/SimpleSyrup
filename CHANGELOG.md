## [1.7.1](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.7.0...v1.7.1) (2026-09-11)


### Bug Fixes

* **regional:** preserve shared model patch ancestry ([6059a3f](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/6059a3f913a9502671666e83faeb8686a7a8da27))

# [1.7.0](https://github.com/Artificial-Sweetener/SimpleSyrup/compare/v1.6.0...v1.7.0) (2026-09-05)


### Bug Fixes

* **anima:** support regional prompting across Comfy versions ([41a234a](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/41a234a85a4cfcdc4cfce68b80b4b9982719aab4))
* **attention:** preserve anchored concept geometry ([1136efd](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/1136efd2ad8708b14320d04eab2f6489efbec282))
* **cache:** make integer narrowing checker-independent ([2547767](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/254776791766c41c75a69ceb5b207c54949446c9))
* **detailers:** align SEGS mask blending behavior ([a3120ae](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/a3120aebe8834982706d93784a39ab501c6ff40a))
* **groundingdino:** support transformers v4 and v5 ([3387c03](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/3387c03eb59f244d56f6a0dfe86cedab1847a8e2))
* **mask:** preserve missing-alpha image geometry ([caf7d37](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/caf7d37a154ddc62405807b56550efdaa831d09e))
* **media:** stabilize native ordered preview controls ([1235652](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/123565225a1104234a7c9f5c43af0772c7508db8))
* **regional:** align prompt batches and LoRA hooks ([656d197](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/656d1970e12be18304b057b07204dcbbe367432b))
* **runtime:** centralize Comfy patcher lifecycle ([0e5f513](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/0e5f513ae0f33f40c6a8bd09161043a5af598392))
* **sampling:** normalize model-specific latent layouts ([b2084a7](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/b2084a7a9bf04709af51380bc1ef4a09ccb9babc))
* **tiled-diffusion:** clamp overlap for small latents ([fecba36](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/fecba36e4e11f0da681c6a5d9d42e18093d741fc))
* **tools:** return host-native checkpoint selections ([c7cb8d2](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/c7cb8d29f83ebd44b48038b7ce5e65a0a6f445b4))
* use SimpleSyrup package identity ([1659d13](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/1659d131f4215fa0baccc4c70024d63590460e67))


### Features

* **anima:** add cached quantization profiles ([c20664f](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/c20664f8925305465ccb4c028d0a78a6364a037d))
* **attention:** add sampler-derived concept regions ([8ca5d2c](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/8ca5d2c325e1caee822883ba568b25f721e51343))
* **attention:** default regional prompts to full weight ([f829321](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/f82932193951bae75d59e1c7c7dba2d187e3c175))
* **attention:** improve concept isolation fidelity and speed ([01826ad](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/01826ad1b7c64b11c8af031691403ecf521cb1b5))
* **attention:** refine attention-derived region masks ([dff84cc](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/dff84cce2322e64b8a9ada91b84550faf3a5c7a1))
* **conditioning:** add regional prompting and SEP-local LoRAs ([d3de028](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/d3de02816b24bea129e77b82568ad47a0fd0ba99))
* **conditioning:** support labeled prompt separators ([a708e0b](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/a708e0b4187b0d5f9aeb58f5ab0d276b8a046395))
* **detailing:** add external llm segs tagging ([207c449](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/207c4492908371f41aca84c74332c1a1f63d8045))
* **detection:** add keep-only SEGS selection ([4f65ae0](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/4f65ae0dd36e43347b40ae64039f40e8a67aea48))
* initial release ([4b6525c](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/4b6525ce6ff42f06a7ffd48a54186fcb625f0e21))
* **loaders:** add automatic FLUX model loaders ([33bff75](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/33bff75af1b001b716a45f4fadc9e0e0aa20ced1))
* **masking:** expand segmentation tooling and progress ([adba198](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/adba1981a4bdb43307496a84cccbfe100ae2174f))
* **media:** add native ordered loaders and SEGS preview ([fcf38f2](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/fcf38f2010cfadc76be74410857387c74db3b575))
* **nodes:** add VAE options and clone-safe diffusion ([5fc0f3d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/5fc0f3d8e5ef5fbac0306b1d3c70464a035c396c))
* **prompt-control:** add schedule and encode prompt node ([af32cec](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/af32cec99f5bdadcbed9f8c33db0d816ad4b72f0))
* **regional:** add native SDXL adapter execution ([068e3db](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/068e3db17d13c875a383c85e6cf931f96459c3b1))
* **regional:** add universal attention coupling foundation ([edfc26c](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/edfc26c9122ff0c18d63e9b80832d587594e9e62))
* **regional:** build universal adapter execution foundation ([864852d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/864852dc9e591a3041e23b342346495a7fbf6589))
* **regional:** complete capability-routed execution ([fe96a20](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/fe96a206dbf1edcae022b1346f998ed184142062))
* **regional:** complete persistent regional LoRA execution ([7b5987d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/7b5987d6fa66f8deee2655d18b1209bbe20271d6))
* **sampling:** add contextual diffusion sampler ([24bf630](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/24bf6309023c552f65947814d9641555a73c8337))
* **sampling:** add deterministic seed variation ([7921be3](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/7921be3f9067a26fe5fa47fb8fb370e7cb1679f3))
* **sampling:** add regional diffusion sampling ([f7dffca](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/f7dffcacfe104be173793ce412f994519e11e02e))
* **sampling:** bypass inactive attention coupling ([b2b8dc4](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/b2b8dc4b016307fd351892f50264c64532898681))
* **sampling:** expose evaluated context SEGS ([04a2c3e](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/04a2c3e6e90f9bbb9e3922412844afe5a4e6869f))
* **segmentation:** add interactive SEGS preview ([c9e303e](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/c9e303ec4727576af124d9e8ea20d0103f67e7ae))
* **segmentation:** add SAM region overlay ([a72c796](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/a72c796d8d6d64a52fed7d281fe53eebc74639ed))
* **segmentation:** add SAM-guided tiled diffusion ([968090d](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/968090de87a4fad726fd37d0a08966c01f11a8fd))
* **segs:** add regional batching and wd14 tagging nodes ([600d9e3](https://github.com/Artificial-Sweetener/SimpleSyrup/commit/600d9e311b5b8c45762e15c88f54df33454261b5))
