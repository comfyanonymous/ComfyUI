# Puteri Gunung Ledang Image Generator (ComfyUI Project)

This project automatically generates four AI-generated anime-style images based on the legendary tale of *Puteri Gunung Ledang*, using [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with a locally hosted server and the `Illustrious-XL-v0.1-GUIDED` checkpoint.

---

## 🌟 Features

- 4 unique image prompts, each based on a different part of the story.
- Uses randomized seed for variety.
- Includes both positive and negative prompts for high-quality generation.
- Images are saved with the filename prefix: `puteri_1`, `puteri_2`, etc.
- Looping mode for repeated generations.

---

## ▶️ How to Use

1. **Start ComfyUI**
   - Make sure ComfyUI is running locally on `http://127.0.0.1:8188`.

2. **Checkpoints & Settings**
   - Confirm that the following model is installed and available:
     ```
     Illustrious-XL-v0.1-GUIDED (1).safetensors
     ```
   - Resolution used: `768x1152`

3. **Run the script**
   ```bash
   python main.py
