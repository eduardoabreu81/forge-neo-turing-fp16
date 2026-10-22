# Turing FP16 for Forge Neo
# Forge Neo picks a model's compute dtype from the dtypes the model lists as supported. Many list only bfloat16 and
# float32 (MOSS-Image, Qwen-Image, Lumina 2, Z-Image, Ernie, PiD, FLUX.1 Schnell), so on a GPU without bfloat16
# (RTX 20 and older) they fall back to float32, several times slower than float16. This hook keeps Forge's choice,
# except that one fallback: there, float16 instead. Same finding as mirabarukaso's ComfyUI_MiraSubPack (MIT), which
# measured MOSS-Image 4.7x (bf16) to 7x (int8) faster on a Titan RTX.
#
# No Forge imports at module level: the CPU tests run it as is.

import functools

import torch

__version__ = "0.1.0"

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


def _enabled() -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, OPTION, DEFAULT))
    except Exception:  # before the options exist
        return DEFAULT


def install() -> bool:
    """Wrap backend.memory_management.inference_cast; False when this Forge Neo lacks what it needs."""
    from backend import memory_management as mm

    if not all(hasattr(mm, n) for n in ("inference_cast", "should_use_bf16", "should_use_fp16")):
        return False
    original = mm.inference_cast
    if getattr(original, "_turing_fp16", False):
        return True

    @functools.wraps(original)
    def inference_cast(weight_dtype, inference_device, *args, **kwargs):
        dtype = original(weight_dtype, inference_device, *args, **kwargs)
        chosen = choose(dtype, weight_dtype, mm.should_use_bf16(inference_device),
                        mm.should_use_fp16(inference_device, prioritize_performance=False), _enabled())
        if chosen != dtype:
            print("[Turing FP16] no bfloat16 on this GPU: computing in float16 instead of float32")
        return chosen

    inference_cast._turing_fp16 = True
    # the loader calls it through the module (memory_management.inference_cast), so the module's name is enough
    mm.inference_cast = inference_cast
    return True


def register_option() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, section=("sd", "Stable Diffusion"),
                                                  category_id="sd").info(INFO))
