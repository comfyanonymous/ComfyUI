import folder_paths
import os
def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    for full_folder_path in full_folder_paths:
        folder_paths.add_model_folder_path(folder_name, full_folder_path)
    if folder_name in folder_paths.folder_names_and_paths:
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        updated_extensions = current_extensions | extensions
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)

image_suffixs = set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg", ".ico", ".apng", ".tif", ".hdr", ".exr"])

model_path = folder_paths.models_dir
add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(model_path, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics_segm", [os.path.join(model_path, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics", [os.path.join(model_path, "ultralytics")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mmdets_bbox", [os.path.join(model_path, "mmdets", "bbox")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mmdets_segm", [os.path.join(model_path, "mmdets", "segm")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mmdets", [os.path.join(model_path, "mmdets")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("sams", [os.path.join(model_path, "sams")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("onnx", [os.path.join(model_path, "onnx")], {'.onnx'})
add_folder_path_and_extensions("instantid", [os.path.join(model_path, "instantid")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("pulid", [os.path.join(model_path, "pulid")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("layer_model", [os.path.join(model_path, "layer_model")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("rembg", [os.path.join(model_path, "rembg")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ipadapter", [os.path.join(model_path, "ipadapter")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("dynamicrafter_models", [os.path.join(model_path, "dynamicrafter_models")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("mediapipe", [os.path.join(model_path, "mediapipe")], set(['.tflite','.pth']))
add_folder_path_and_extensions("inpaint", [os.path.join(model_path, "inpaint")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("prompt_generator", [os.path.join(model_path, "prompt_generator")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("t5", [os.path.join(model_path, "t5")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("llm", [os.path.join(model_path, "LLM")], folder_paths.supported_pt_extensions)