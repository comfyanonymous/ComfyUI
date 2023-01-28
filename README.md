ComfyUI
=======
A powerful and modular stable diffusion GUI.
-----------
![ComfyUI Screenshot](comfyui_screenshot.png)

This ui will let you design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface.


# Installing

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae

At the time of writing this pytorch has issues with python versions higher than 3.10 so make sure your python/pip versions are 3.10.

### AMD
AMD users can install rocm and pytorch with pip if you don't have it already installed:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2```

### NVIDIA

Nvidia users should install torch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117```

Nvidia users should also install Xformers for a speed boost but can still run the software without it.

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```



# Running

```python main.py```

### For AMD 6700, 6600 and maybe others

Try running it with this command if you have issues:

```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

# Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

### Colab Notebook

To run it on colab you can use my [Colab Notebook](notebooks/comfyui_colab.ipynb) here: [Link to open with google colab](https://colab.research.google.com/github/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb)

### Fedora

To get python 3.10 on fedora:
```dnf install python3.10```

Then you can:

```python3.10 -m ensurepip```

This will let you use: pip3.10 to install all the dependencies.

## Support and dev channel

[Matrix room: #comfyui:matrix.org](https://app.element.io/#/room/%23comfyui%3Amatrix.org) (it's like discord but open source).

# QA

### Why did you make this?

I wanted to learn how Stable Diffusion worked in detail. I also wanted something clean and powerful that would let me experiment with SD without restrictions.

### Who is this for?

This is for anyone that wants to make complex workflows with SD or that wants to learn more how SD works. The interface follows closely how SD works and the code should be much more simple to understand than other SD UIs.
