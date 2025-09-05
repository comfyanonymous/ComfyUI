from .cfz_caching_condition import save_conditioning, load_conditioning, CFZ_PrintMarker

NODE_CLASS_MAPPINGS = {
    "CFZ_save_conditioning": save_conditioning,
    "CFZ_load_conditioning": load_conditioning,
    "CFZ_PrintMarker": CFZ_PrintMarker,   # NEW
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFZ_save_conditioning": "CFZ Save Conditioning",
    "CFZ_load_conditioning": "CFZ Load Conditioning",
    "CFZ_PrintMarker": "CFZ Print Marker",
}
