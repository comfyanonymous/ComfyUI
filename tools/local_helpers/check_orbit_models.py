
import os

root_dir = r"c:\Users\jando\work\ComfyUI"
required_files = [
    "juggerxlInpaint_juggerInpaintV8.safetensors",
    "pulid_flux_v0.9.1.safetensors",
    "flux1-dev-Q4_K_S.gguf",
    "ae.sft",
    "inswapper_128.onnx",
    "retinaface_resnet50.onnx", 
    "GPEN-BFR-512.onnx",
    # Note: "RealisticVision Flux Dev_Flux_Dev_FP8.safetensors" seems to be a mix, 
    # likely "RealisticVision" checks point or Flux Dev FP8? 
    # The JSON says: "RealisticVision Flux Dev_Flux_Dev_FP8.safetensors"
    "RealisticVision Flux Dev_Flux_Dev_FP8.safetensors",
    "ip-adapter.bin",
    "clip_l.safetensors",
    "t5xxl_fp8_e4m3fn.safetensors",
    "inswapper_128_fp16.onnx"
]

found_files = {}

def scan_dir(directory):
    print(f"Scanning {directory}...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file in required_files:
                path = os.path.join(root, file)
                print(f"FOUND: {file} at {path}")
                found_files[file] = path

if __name__ == "__main__":
    scan_dir(os.path.join(root_dir, "models"))
    scan_dir(os.path.join(root_dir, "custom_nodes")) # Sometimes models are hidden in nodes
    
    print("\n--- Summary ---")
    missing = [f for f in required_files if f not in found_files]
    if missing:
        print("MISSING FILES:")
        for m in missing:
            print(f" - {m}")
    else:
        print("All files found!")
