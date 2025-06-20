# Auto Image Generator – Puteri Gunung Ledang (with filename_prefix)

This Python script is part of a digital project inspired by the Malaysian legend of Puteri Gunung Ledang, a mystical Malay princess known for her wisdom, strength, and unattainability. The script automatically sends 7 prompts to ComfyUI API to generate anime-style images based on this legend.

---

## 🧝‍♀️ About the Project

This project reimagines Puteri Gunung Ledang, the legendary princess who resides on Gunung Ledang, through the lens of modern AI-generated art. The 7 image prompts reflect key parts of the legend, including her mystical beauty and the seven impossible conditions she set for the Sultan of Melaka.

We generated the images using ComfyUI with a customized workflow and converted them into a short video.

🎥 Video Showcase:  
📎 *Watch the final video here* (replace with actual link)

---

## 👨‍👩‍👧‍👦 Team Info

- Team Name: Hello Kitty  
- Competition: Young Digital Innovators (Python Category)  
- GitHub Repo: https://github.com/PenangScienceCluster/python2025

---

## 🛠 Tools & Resources

- 🔧 ComfyUI (used for image generation):  
  https://github.com/comfyanonymous/ComfyUI

- 🧠 Model used: Illustrious-XL v0.1 (GUIDED)  
  Trained by Onoma AI  
  Download from HuggingFace:  
  https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/main/Illustrious-XL-v0.1-GUIDED.safetensors

- ✂️ Video Editing Tool: 必剪 (BiJian)

---

## 🔧 Customizations in This Project

- Forked and ran ComfyUI locally
- Integrated the Illustrious-XL-v0.1-GUIDED model manually
- Designed custom workflows (`puteri_1.json` to `puteri_7.json`) tailored to different scenes
- Wrote a Python script (`main.py`) that:
  - Automatically loads 7 prompt workflows
  - Sends them to ComfyUI API (localhost)
  - Sets seed, saves images with filename_prefix
- All prompts are structured to reflect the seven legendary conditions or scenes in the story

---

## ✅ Requirements

- ComfyUI installed and running on: http://127.0.0.1:8188  
  → [Install instructions](https://github.com/comfyanonymous/ComfyUI)
- Python 3.12
- Dependencies:
  ```bash
  pip install requests
  
---

🚀 How to Start
1. Launch ComfyUI
git clone https://github.com/weiyangfoh/ComfyUI
cd ComfyUI
python main.py
Make sure the Illustrious XL model is placed in your models/checkpoints folder.

3. Run Our Script
cd Group-Hello-Kitty
In the Group-Hello-Kitty folder:
python main.py
It will automatically call ComfyUI to generate 7 images with the correct filename prefix.
Images will be saved to your ComfyUI /output/ folder.
