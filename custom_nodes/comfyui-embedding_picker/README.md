# Embedding Picker

Tired of forgetting and misspelling often weird names of embeddings you use?
Or perhaps you use only one, cause you forgot you have tens of them installed?

Try this.

# Installation

Exactly the same as with other simple custom nodes.

- Click the green **Code** button, select **Download Zip**, and unpack it in your ComfyUI `custom_nodes` directory

or

- Clone this repository by running `git clone https://github.com/Tropfchen/ComfyUI-Embedding_Picker.git` in your ComfyUI `custom_nodes` directory

To uninstall:

- Delete `ComfyUi_Embedding_Picker` in your ComfyUI custom_nodes directory

# Use

Right click on the _CLIP Text Encode_ node and select the top option 'Prepend Embedding Picker'. This will create the node itself and copy all your prompts.
A similar option exists on the `Embedding Picker' node itself, use this to quickly chain multiple embeddings.

You can also find the node in `utils` menu.

By default the embedding will be the placed as first prompt, and will result in such text

> embedding:EasyNegative.pt, text, watermark

Set `append` to True if you prefer it to be at the end, i. e.

> text, watermark, embedding:EasyNegative.pt

## Example Workflow

![Example](example_workflow.png)
