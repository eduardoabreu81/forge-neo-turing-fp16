# ⚡ Turing FP16 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](https://github.com/eduardoabreu81/forge-neo-turing-fp16)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Faster generation on GPUs without bfloat16 (RTX 20 series, Titan RTX, GTX 16 and older): models that Forge Neo would run in float32 run in float16 instead.

<p align="center"><img src=".github/images/moss-fp32-vs-fp16.jpg" alt="The same MOSS-Image portrait made in float32 (33 seconds) and in float16 with the extension (16 seconds) on an RTX 2070" width="720"></p>

> [!Note]
> Tested on an RTX 2070 (8 GB), Forge Neo's own MOSS-Image, 8 steps, 1024x1024, the same picture up to small details:
> - INT8 (SFT with the 8-step LoRA, or the hypermosslush turbo merge): 32-33 to 16 seconds a picture
> - BF16 (Refined Realism v2 merge): 78 to 24 seconds a picture
>
> Other models have not been tested yet.

---

## 🎯 What It Does

- Forge Neo picks each model's compute precision from the ones the model lists. Many list only bfloat16 and float32 - MOSS-Image, Qwen-Image, Lumina 2, Z-Image, Ernie, PiD, FLUX.1 Schnell - so a GPU without bfloat16 runs them in float32, several times slower
- With this extension, that one fallback becomes float16; every other choice of Forge Neo stays as it is
- On a card with room for the float32 weights (Titan RTX, RTX 2080 Ti), Forge Neo also widens BF16 files to float32 in memory; there they keep their own bfloat16 weights instead, half the memory
- [mirabarukaso's ComfyUI_MiraSubPack](https://github.com/mirabarukaso/ComfyUI_MiraSubPack) measured the same switch on MOSS-Image: 4.7x faster with the BF16 files and about 7x with the INT8 ones, on a Titan RTX
- Turn it off under **Settings → Stable Diffusion → [Turing FP16]**: a model whose numbers outgrow float16 can come out black or burnt

---

## 📦 Installation

1. Go to **Extensions** → **Install from URL**
2. Paste: `https://github.com/eduardoabreu81/forge-neo-turing-fp16`
3. Click **Install** and restart the WebUI

The console says `[Turing FP16] no bfloat16 on this GPU: computing in float16 instead of float32` when it switches a model.

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[ComfyUI_MiraSubPack](https://github.com/mirabarukaso/ComfyUI_MiraSubPack)** by mirabarukaso - the finding and the measurements

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)
