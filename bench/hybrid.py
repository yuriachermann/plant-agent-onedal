"""Hybrid CPUs on Windows: core classes, where a thread runs, and the EcoQoS opt-out.

Windows can throttle a process with EcoQoS, and a throttled process tends to run
on the E-cores. In six probes on a 12th Gen laptop, the main thread of a process
that did not opt out ran mostly on the E-cores in 10 of 12 processes. With the
opt-out it stayed on the P-cores in all 12 (docs/findings/stock-gap.md, section 6).

The benchmark processes (``bench.run_bench`` and ``bench.kernels``) opt out unless
``PLANT_BENCH_ECOQOS=keep``. Everything here is a no-op off Windows.
"""
from __future__ import annotations

import os
import platform
import sys

ECOQOS_ENV = "PLANT_BENCH_ECOQOS"


def cpu_sets() -> list[dict]:
    """Logical processors with their core and efficiency class. Empty off Windows."""
    if platform.system() != "Windows":
        return []
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetSystemCpuSetInformation
    fn.restype = wintypes.BOOL
    fn.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG), wintypes.HANDLE, wintypes.ULONG]
    need = wintypes.ULONG(0)
    fn(None, 0, ctypes.byref(need), None, 0)
    if need.value == 0:
        return []
    buf = (ctypes.c_ubyte * need.value)()
    if not fn(buf, need, ctypes.byref(need), None, 0):
        return []
    raw = bytes(buf)
    out, off = [], 0
    while off < len(raw):
        size = int.from_bytes(raw[off:off + 4], "little")
        if size == 0:
            break
        out.append({"logical": raw[off + 14], "core": raw[off + 15], "efficiency_class": raw[off + 18]})
        off += size
    return out


_GET_CPU = None


def current_cpu() -> int | None:
    """Logical processor the calling thread is on right now. None off Windows."""
    global _GET_CPU
    if platform.system() != "Windows":
        return None
    if _GET_CPU is None:
        import ctypes
        _GET_CPU = ctypes.WinDLL("kernel32").GetCurrentProcessorNumber
    return int(_GET_CPU())


def placement(cpus: list[int | None], sets: list[dict]) -> dict:
    """Share of main-thread samples on the fastest efficiency class (the P-cores).

    ``pcore_share`` is None when there are no samples or the CPU has one core class.
    """
    cls = {s["logical"]: s["efficiency_class"] for s in sets}
    seen = [cls[c] for c in cpus if c is not None and c in cls]
    if not seen or len(set(cls.values())) < 2:
        return {"samples": len(seen), "pcore_share": None}
    hi = max(cls.values())
    return {"samples": len(seen), "pcore_share": round(sum(c == hi for c in seen) / len(seen), 3)}


def opt_out_of_ecoqos() -> None:
    """Ask Windows not to throttle this process's execution speed (EcoQoS). Affects this process only."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.restype = wintypes.BOOL
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    process_power_throttling, execution_speed = 4, 0x1
    st = State(1, execution_speed, 0)  # control execution speed, state off = never throttle
    if not k32.SetProcessInformation(k32.GetCurrentProcess(), process_power_throttling, ctypes.byref(st),
                                     ctypes.sizeof(st)):
        raise OSError(ctypes.get_last_error(), "SetProcessInformation(ProcessPowerThrottling) failed")


def apply_ecoqos_policy() -> bool | None:
    """Opt this benchmark process out of EcoQoS unless ``PLANT_BENCH_ECOQOS=keep``.

    Returns True when the process opted out, False when it kept the Windows default
    (asked to, or the call failed), and None off Windows.
    """
    if platform.system() != "Windows":
        return None
    if os.environ.get(ECOQOS_ENV, "").strip().lower() == "keep":
        return False
    try:
        opt_out_of_ecoqos()
    except OSError as exc:
        print(f"EcoQoS opt-out failed, timings use the Windows default: {exc}", file=sys.stderr)
        return False
    return True
