import torch

from lib_turing_fp16 import choose, choose_storage

F16, BF16, F32 = torch.float16, torch.bfloat16, torch.float32


def test_float32_fallback_becomes_float16_without_bf16():
    assert choose(F32, BF16, bf16_ok=False, fp16_ok=True, enabled=True) == F16
    assert choose(F32, "int8", bf16_ok=False, fp16_ok=True, enabled=True) == F16


def test_forge_choice_kept_otherwise():
    assert choose(F32, BF16, bf16_ok=True, fp16_ok=True, enabled=True) == F32  # a GPU with bf16
    assert choose(F32, BF16, bf16_ok=False, fp16_ok=False, enabled=True) == F32  # no fp16 either
    assert choose(F32, BF16, bf16_ok=False, fp16_ok=True, enabled=False) == F32  # turned off
    assert choose(F32, F32, bf16_ok=False, fp16_ok=True, enabled=True) == F32  # stored in fp32 on purpose
    assert choose(BF16, BF16, bf16_ok=True, fp16_ok=True, enabled=True) == BF16
    assert choose(F16, F16, bf16_ok=False, fp16_ok=True, enabled=True) == F16


def test_half_precision_file_keeps_its_storage_without_bf16():
    assert choose_storage(F32, BF16, bf16_ok=False, fp16_ok=True, enabled=True) == BF16
    assert choose_storage(F32, F16, bf16_ok=False, fp16_ok=True, enabled=True) == F16
    # and then computes in float16
    assert choose(F32, BF16, bf16_ok=False, fp16_ok=True, enabled=True) == F16


def test_forge_storage_kept_otherwise():
    assert choose_storage(F32, BF16, bf16_ok=True, fp16_ok=True, enabled=True) == F32  # a GPU with bf16
    assert choose_storage(F32, BF16, bf16_ok=False, fp16_ok=False, enabled=True) == F32  # no fp16 either
    assert choose_storage(F32, BF16, bf16_ok=False, fp16_ok=True, enabled=False) == F32  # turned off
    assert choose_storage(F32, BF16, bf16_ok=False, fp16_ok=True, enabled=True, forced_fp32=True) == F32  # --fp32-unet
    assert choose_storage(F32, F32, bf16_ok=False, fp16_ok=True, enabled=True) == F32  # a float32 file
    assert choose_storage(F16, BF16, bf16_ok=False, fp16_ok=True, enabled=True) == F16  # Forge's own choice
