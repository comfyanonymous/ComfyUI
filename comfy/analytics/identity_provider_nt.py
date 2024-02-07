import ctypes
from ctypes import wintypes, POINTER, byref

_windows_dll = ctypes.WinDLL('Secur32.dll')

_windows_get_user_name_ex_w_func = _windows_dll.GetUserNameExW
_windows_get_user_name_ex_w_func.argtypes = [ctypes.c_int, POINTER(wintypes.WCHAR), POINTER(wintypes.ULONG)]
_windows_get_user_name_ex_w_func.restype = wintypes.BOOL

_windows_extended_name_format = {
    "NameUnknown": 0,
    "NameFullyQualifiedDN": 1,
    "NameSamCompatible": 2,
    "NameDisplay": 3,
    "NameUniqueId": 6,
    "NameCanonical": 7,
    "NameUserPrincipal": 8,
    "NameCanonicalEx": 9,
    "NameServicePrincipal": 10,
    "NameDnsDomain": 12
}


def get_user_name():
    size = wintypes.ULONG(0)
    format_type = _windows_extended_name_format["NameDisplay"]
    _windows_get_user_name_ex_w_func(format_type, None, byref(size))

    name_buffer = ctypes.create_unicode_buffer(size.value)

    if not _windows_get_user_name_ex_w_func(format_type, name_buffer, byref(size)):
        return None

    return name_buffer.value
