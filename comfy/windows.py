import ctypes
import logging
import psutil
from ctypes import wintypes

import comfy_aimdo.control

psapi = ctypes.WinDLL("psapi")
kernel32 = ctypes.WinDLL("kernel32")

class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]

def get_free_ram():
    #Windows is way too conservative and chalks recently used uncommitted model RAM
    #as "in-use". So, calculate free RAM for the sake of general use as the greater of:
    #
    #1: What psutil says
    #2: Total Memory - (Committed Memory - VRAM in use)
    #
    #We have to subtract VRAM in use from the comitted memory as WDDM creates a naked
    #commit charge for all VRAM used just incase it wants to page it all out. This just
    #isn't realistic so "overcommit" on our calculations by just subtracting it off.

    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)

    if not psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        logging.warning("WARNING: Failed to query windows performance info. RAM usage may be sub optimal")
        return psutil.virtual_memory().available

    committed = pi.CommitTotal * pi.PageSize
    total = pi.PhysicalTotal * pi.PageSize

    return max(psutil.virtual_memory().available,
               total - (committed - comfy_aimdo.control.get_total_vram_usage()))

