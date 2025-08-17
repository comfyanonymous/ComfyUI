# Florence2 in ComfyUI

> Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. 
Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. 
It leverages our FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. 
The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

## New Feature: Document Visual Question Answering (DocVQA)

This fork includes support for Document Visual Question Answering (DocVQA) using the Florence2 model. DocVQA allows you to ask questions about the content of document images, and the model will provide answers based on the visual and textual information in the document. This feature is particularly useful for extracting information from scanned documents, forms, receipts, and other text-heavy images.

## Installation:

Clone this repository to 'ComfyUI/custom_nodes` folder.

Install the dependencies in requirements.txt, transformers version 4.38.0 minimum is required:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Florence2\requirements.txt`

![image](https://github.com/kijai/ComfyUI-Florence2/assets/40791699/4d537ac7-5490-470f-92f5-3007da7b9cc7)
![image](https://github.com/kijai/ComfyUI-Florence2/assets/40791699/512357b7-39ee-43ee-bb63-7347b0a8d07d)

Supports most Florence2 models, which can be automatically downloaded with the `DownloadAndLoadFlorence2Model` to `ComfyUI/models/LLM`:

Official:

https://huggingface.co/microsoft/Florence-2-base

https://huggingface.co/microsoft/Florence-2-base-ft

https://huggingface.co/microsoft/Florence-2-large

https://huggingface.co/microsoft/Florence-2-large-ft

https://huggingface.co/HuggingFaceM4/Florence-2-DocVQA

Tested finetunes:

https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5

https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5

https://huggingface.co/thwri/CogFlorence-2.2-Large

https://huggingface.co/HuggingFaceM4/Florence-2-DocVQA

https://huggingface.co/gokaygokay/Florence-2-SD3-Captioner

https://huggingface.co/gokaygokay/Florence-2-Flux-Large

https://huggingface.co/NikshepShetty/Florence-2-pixelpros

## Using DocVQA

To use the DocVQA feature:
1. Load a document image into ComfyUI.
2. Connect the image to the Florence2 DocVQA node.
3. Input your question about the document.
4. The node will output the answer based on the document's content.

Example questions:
- "What is the total amount on this receipt?"
- "What is the date mentioned in this form?"
- "Who is the sender of this letter?"

Note: The accuracy of answers depends on the quality of the input image and the complexity of the question.
