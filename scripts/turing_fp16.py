# Turing FP16 for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before any model loads. Any failure leaves Forge Neo
# unchanged and says so in the console.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import lib_turing_fp16

    if lib_turing_fp16.install():
        lib_turing_fp16.register_option()
        print("[Turing FP16] enabled")
    else:
        print("[Turing FP16] this Forge Neo lacks what the extension needs, so it stays disabled")
except Exception:
    print("[Turing FP16] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()
