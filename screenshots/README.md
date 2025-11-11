# Screenshots for PR Submission

This directory should contain screenshots demonstrating the enhanced `run_comfyui.bat` features.

## Required Screenshots

### High Priority

1. **ASCII Art Banner** (`01_ascii_banner.png`)
   - Capture the ASCII art banner showing "Comfy" text
   - Shows the polished, professional appearance of the script
   - Capture right after running the script

2. **Dependency Checking Prompt** (`02_dependency_check.png`)
   - Capture the prompt showing missing dependencies with installation options
   - Demonstrates the automated dependency checking feature
   - Capture when critical dependencies are missing

3. **CUDA PyTorch Detection** (`03_cuda_detection.png`)
   - Capture the CPU-only PyTorch detection message and installation offer
   - Shows the automatic CUDA PyTorch detection and installation feature
   - Capture when CPU-only PyTorch is detected

### Medium Priority

4. **Progress Bar During Installation** (`04_progress_bar.png`)
   - Capture progress bar showing during pip installation (especially PyTorch)
   - Demonstrates the progress bar feature for long installations
   - Capture during pip install with `--progress-bar on`

5. **Virtual Environment Detection** (`05_venv_detection.png`)
   - Capture message showing virtual environment detection
   - Shows the virtual environment awareness feature
   - Capture when running in a virtual environment

### Low Priority

6. **Error Message Example** (`06_error_message.png`)
   - Capture one of the user-friendly error messages with troubleshooting steps
   - Demonstrates improved error handling
   - Capture when an error occurs (e.g., Python not found)

## How to Capture Screenshots

1. Run `run_comfyui.bat` in various scenarios
2. Use Windows Snipping Tool (Win + Shift + S) or Print Screen
3. Save screenshots with the naming convention above
4. Add screenshots to this directory
5. Update `PR_DESCRIPTION.md` to reference these screenshots

## Note

Screenshots are optional but highly recommended for PR submission. They help reviewers understand the user experience improvements.

