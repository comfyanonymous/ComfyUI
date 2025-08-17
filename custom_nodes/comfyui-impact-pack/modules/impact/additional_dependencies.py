import sys
import subprocess


def ensure_onnx_package():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
            subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', 'onnxruntime'])
        else:
            subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', 'onnxruntime'])
