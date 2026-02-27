import os
from setuptools import setup, find_packages

setup(
    name="comfyui_frontend_package",
    version=os.getenv("COMFYUI_FRONTEND_VERSION") or "0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.9",
)
