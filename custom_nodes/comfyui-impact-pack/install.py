import os
import shutil
import sys
import subprocess
import threading
import locale
import traceback


if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version


impact_path = os.path.join(os.path.dirname(__file__), "modules")


comfy_path = os.environ.get('COMFYUI_PATH')
if comfy_path is None:
    print(f"\nWARN: The `COMFYUI_PATH` environment variable is not set. Assuming `{os.path.dirname(__file__)}/../../` as the ComfyUI path.", file=sys.stderr)
    comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

model_path = os.environ.get('COMFYUI_MODEL_PATH')
if model_path is None:
    try:
        import folder_paths
        model_path = folder_paths.models_dir
    except:
        pass

    if model_path is None:
        model_path = os.path.abspath(os.path.join(comfy_path, 'models'))
    print(f"\nWARN: The `COMFYUI_MODEL_PATH` environment variable is not set. Assuming `{model_path}` as the ComfyUI path.", file=sys.stderr)


sys.path.append(impact_path)
sys.path.append(comfy_path)


# ---
def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else: 
            print(msg, end="", file=sys.stderr)
            

def process_wrap(cmd_str, cwd=None, handler=None, env=None):
    print(f"[Impact Pack] EXECUTE: {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()
# ---


try:
    from torchvision.datasets.utils import download_url
    import impact.config

    print("### ComfyUI-Impact-Pack: Check dependencies")
    def install():
        new_env = os.environ.copy()
        new_env["COMFYUI_PATH"] = comfy_path
        new_env["COMFYUI_MODEL_PATH"] = model_path

        # Download model
        print("### ComfyUI-Impact-Pack: Check basic models")
        sam_path = os.path.join(model_path, "sams")
        onnx_path = os.path.join(model_path, "onnx")

        if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'skip_download_model')):
            try:
                if not os.path.exists(os.path.join(sam_path, "sam_vit_b_01ec64.pth")):
                    download_url("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_path)
            except:
                print("[Impact Pack] Failed to auto-download model files. Please download them manually.")

        if not os.path.exists(onnx_path):
            print(f"### ComfyUI-Impact-Pack: onnx model directory created ({onnx_path})")
            os.mkdir(onnx_path)

        impact.config.write_config()

        # Remove legacy subpack
        try:
            subpack_path = os.path.join(os.path.dirname(__file__), 'impact_subpack')
            if os.path.exists(subpack_path):
                shutil.rmtree(subpack_path)
                print(f"Legacy subpack is detected. '{subpack_path}' is removed.")
                
            subpack_path = os.path.join(os.path.dirname(__file__), 'subpack')
            if os.path.exists(subpack_path):
                shutil.rmtree(subpack_path)
                print(f"Legacy subpack is detected. '{subpack_path}' is removed.")
        except:
            print(f"ERROT: Failed to delete legacy subpack '{subpack_path}'\nPlease delete the folder after terminate ComfyUI.")

    install()

except Exception:
    print("[ERROR] ComfyUI-Impact-Pack: Dependency installation has failed. Please install manually.")
    traceback.print_exc()
