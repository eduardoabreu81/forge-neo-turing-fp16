# Turing FP16 for Forge Neo
# Forge Neo picks a model's compute dtype from the dtypes the model lists as supported. Many list only bfloat16 and
# float32 (MOSS-Image, Qwen-Image, Lumina 2, Z-Image, Ernie, PiD, FLUX.1 Schnell), so on a GPU without bfloat16
# (RTX 20 and older) they fall back to float32, several times slower than float16. This hook keeps Forge's choice,
# except that one fallback: there, float16 instead. For the same reason Forge stores a bfloat16 file in float32; it
# keeps its own bfloat16 weights instead, half the memory, and computes in float16. Same finding as mirabarukaso's ComfyUI_MiraSubPack (MIT), which
# measured MOSS-Image 4.7x (bf16) to 7x (int8) faster on a Titan RTX.
#
# No Forge imports at module level: the CPU tests run it as is.

import functools

import torch

__version__ = "0.2.0"

OPTION = "turing_fp16_enable"
DEFAULT = True
LABEL = "[Turing FP16] Compute in float16 where Forge would fall back to float32 on a GPU without bfloat16"
INFO = "RTX 20 and older; a model whose numbers outgrow float16 can turn out black or burnt - turn this off for it"


def choose(original_dtype: torch.dtype, weight_dtype, bf16_ok: bool, fp16_ok: bool, enabled: bool) -> torch.dtype:
    """The compute dtype: Forge's, or float16 when Forge fell back to float32 only for lack of bfloat16."""
    if not enabled or original_dtype != torch.float32 or bf16_ok or not fp16_ok:
        return original_dtype
    # a model stored in float32 on purpose keeps it
    if weight_dtype == torch.float32:
        return original_dtype
    return torch.float16


def choose_storage(original_dtype: torch.dtype, weight_dtype, bf16_ok: bool, fp16_ok: bool, enabled: bool,
                   forced_fp32: bool = False) -> torch.dtype:
    """The storage dtype: Forge's, or the file's own half precision when Forge widened it to float32 only for lack of
    bfloat16. Without this, a bfloat16 file is stored in float32 and choose() can no longer tell it from a model kept in
    float32 on purpose."""
    if not enabled or forced_fp32 or original_dtype != torch.float32 or bf16_ok or not fp16_ok:
        return original_dtype
    if weight_dtype in (torch.bfloat16, torch.float16):
        return weight_dtype
    return original_dtype


def _enabled() -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, OPTION, DEFAULT))
    except Exception:  # before the options exist
        return DEFAULT


def install() -> bool:
    """Wrap backend.memory_management.unet_dtype and inference_cast; False when this Forge Neo lacks what it needs."""
    from backend import memory_management as mm

    if not all(hasattr(mm, n) for n in ("unet_dtype", "inference_cast", "should_use_bf16", "should_use_fp16")):
        return False
    if getattr(mm.inference_cast, "_turing_fp16", False):
        return True

    original_storage = mm.unet_dtype
    original_cast = mm.inference_cast

    @functools.wraps(original_storage)
    def unet_dtype(*args, **kwargs):
        dtype = original_storage(*args, **kwargs)
        device = kwargs.get("device", args[0] if args else None)
        weight_dtype = kwargs.get("weight_dtype", args[3] if len(args) > 3 else None)
        chosen = choose_storage(dtype, weight_dtype, mm.should_use_bf16(device),
                                mm.should_use_fp16(device, prioritize_performance=False), _enabled(),
                                forced_fp32=bool(getattr(mm.args, "fp32_unet", False)))
        if chosen != dtype:
            print(f"[Turing FP16] no bfloat16 on this GPU: keeping the {str(chosen)[6:]} weights instead of float32")
        return chosen

    @functools.wraps(original_cast)
    def inference_cast(weight_dtype, inference_device, *args, **kwargs):
        dtype = original_cast(weight_dtype, inference_device, *args, **kwargs)
        chosen = choose(dtype, weight_dtype, mm.should_use_bf16(inference_device),
                        mm.should_use_fp16(inference_device, prioritize_performance=False), _enabled())
        if chosen != dtype:
            print("[Turing FP16] no bfloat16 on this GPU: computing in float16 instead of float32")
        return chosen

    unet_dtype._turing_fp16 = inference_cast._turing_fp16 = True
    # the loader calls both through the module (memory_management.unet_dtype), so the module's names are enough
    mm.unet_dtype = unet_dtype
    mm.inference_cast = inference_cast
    return True


def register_option() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, section=("sd", "Stable Diffusion"),
                                                  category_id="sd").info(INFO))
