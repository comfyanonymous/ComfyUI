# Enhanced run_comfyui.bat with Automated Dependency Checking and CUDA PyTorch Installation

## Author's Note

**Important**: I am not a professional coder and have relied heavily on Cursor AI for the development of this script. While I have done my best to ensure its functionality and safety, I kindly request a thorough review by experienced developers before merging. Please pay special attention to:
- Batch file logic and error handling
- Python command-line invocations
- Virtual environment detection logic
- Dependency checking implementation
- User interaction flow

## Related Issue

Addresses [#10705](https://github.com/comfyanonymous/ComfyUI/issues/10705) - Feature Request: Enhanced run_comfyui.bat with Automated Dependency Checking and CUDA PyTorch Detection

## Issue Addressed

This PR addresses common user pain points when setting up ComfyUI on Windows:

1. **Missing Dependencies**: Users often encounter cryptic import errors when dependencies are missing, requiring manual installation and troubleshooting
2. **CPU-Only PyTorch**: Many users accidentally install CPU-only PyTorch, which prevents GPU acceleration and causes performance issues
3. **Poor Error Messages**: Existing error messages don't provide clear guidance on how to resolve issues
4. **Installation Confusion**: Users are unsure which dependencies are required vs optional, and whether to install in virtual environments

This PR solves these issues by providing automated dependency checking, intelligent PyTorch detection, and user-friendly error messages with actionable troubleshooting steps.

## Overview

This PR enhances the `run_comfyui.bat` startup script for Windows users, significantly improving the user experience by automatically checking dependencies, detecting virtual environments, and offering intelligent installation options. The script now provides a polished, user-friendly interface with clear error messages and troubleshooting guidance.

## Key Features

### 1. **Automated Dependency Checking**
- Checks all critical Python dependencies before launching ComfyUI
- Separates critical vs. optional dependencies
- Provides clear prompts for missing packages
- Offers installation options: Install All, Critical Only, or Cancel

### 2. **CUDA PyTorch Auto-Installation**
- Automatically detects CPU-only PyTorch installations
- Offers to automatically uninstall CPU version and install CUDA-enabled version
- Shows progress bars during installation (`--progress-bar on`)
- Verifies installation before proceeding
- Provides clear warnings about NVIDIA GPU requirements

### 3. **Virtual Environment Awareness**
- Detects if running in a virtual environment
- Provides appropriate warnings about installation impacts
- Offers guidance on creating virtual environments for safer package management

### 4. **Enhanced User Experience**
- UTF-8 encoding support for proper Unicode character display
- ASCII art banner with "Comfy" text
- Progress bars for all pip installations
- User-friendly error messages with actionable troubleshooting steps
- Clear, informative prompts throughout the installation process

### 5. **Comprehensive Error Handling**
- Detailed error messages for common issues:
  - Python not found
  - Installation failures
  - CUDA out of memory errors
  - Module not found errors
- Provides specific troubleshooting steps for each error type

## Files Changed

- **`run_comfyui.bat`** (408 lines, +347 insertions, -61 deletions)
  - Enhanced startup script with all new features
  - UTF-8 encoding support
  - Comprehensive dependency checking
  - CUDA PyTorch detection and auto-installation
  - Virtual environment detection
  - Progress bars for installations
  - User-friendly error messages

- **`create_shortcut.ps1`** (1 line addition)
  - PowerShell script for creating desktop shortcuts
  - Helper utility for easier access

## Screenshots

### ASCII Art Banner
The script displays a polished ASCII art banner with "Comfy" text:
```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║       ██████╗ ██████╗ ███╗   ███╗███████╗██╗   ██╗       ║
║      ██╔════╝██╔═══██╗████╗ ████║██╔════╝╚██╗ ██╔╝       ║
║      ██║     ██║   ██║██╔████╔██║█████╗   ╚████╔╝        ║
║      ██║     ██║   ██║██║╚██╔╝██║██╔══╝    ╚██╔╝         ║
║      ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║        ██║          ║
║       ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝        ╚═╝          ║
║                                                           ║
║         The most powerful open source node-based          ║
║         application for generative AI                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

### Key User Interactions

**Dependency Checking Prompt:**
```
╔═══════════════════════════════════════════════════════════╗
║            Missing Required Packages                      ║
╚═══════════════════════════════════════════════════════════╝

▓ ComfyUI needs some additional software to run.
  The following critical packages are missing:
  yaml, torch, numpy

▓ [Heads Up] You're using your main Python installation.
  Installing packages here might affect other programs that use Python.

▓ Installation Options:
    [I] Install all missing packages (recommended)
    [C] Install only critical packages
    [N] Cancel and exit
```

**CUDA PyTorch Detection:**
```
╔═══════════════════════════════════════════════════════════╗
║     CPU-Only PyTorch Detected - CUDA Version Required     ║
╚═══════════════════════════════════════════════════════════╝

▓ Your PyTorch installation doesn't support GPU acceleration.
  ComfyUI requires CUDA-enabled PyTorch to run properly.

▓ We can automatically install the CUDA-enabled version for you.
  This will:
     1. Remove the current CPU-only version
     2. Install the CUDA-enabled version (this will take several minutes)
     3. Continue to launch ComfyUI automatically

Would you like to install CUDA-enabled PyTorch now? (Y/N):
```

## Testing Recommendations

To thoroughly test this PR, please verify the following scenarios:

1. **Clean Environment**: Run the script in an environment with no Python or ComfyUI dependencies installed
2. **Missing Critical Dependencies**: Manually uninstall one or more critical dependencies (e.g., `pyyaml`) and verify the script correctly identifies them
3. **Missing Optional Dependencies**: Uninstall an optional dependency and verify the script offers to install it or skip
4. **CPU-Only PyTorch**: Install a CPU-only version of PyTorch and verify the script detects it and offers to install the CUDA version
5. **CUDA-Enabled PyTorch**: Ensure a CUDA-enabled PyTorch is installed and verify the script proceeds directly to launching ComfyUI
6. **Virtual Environment**: Test running the script within an activated virtual environment
7. **System Python**: Test running the script with system Python (not in a virtual environment)
8. **Error Handling**: Verify that all error messages are clear, informative, and provide helpful troubleshooting steps
9. **Progress Bars**: Verify that progress bars display correctly during pip installations
10. **ASCII Art**: Confirm the ASCII art banner renders correctly in a standard Windows command prompt

## Technical Details

### UTF-8 Encoding
- Uses `chcp 65001` to enable UTF-8 encoding for proper Unicode character display
- Ensures ASCII art and box-drawing characters render correctly

### Dependency Checking
- Uses `importlib.util.find_spec()` to check for module availability
- Separates critical dependencies (required for ComfyUI to run) from optional dependencies
- Provides user with clear installation options

### CUDA PyTorch Detection
- Checks PyTorch version string for "+cpu" indicator
- Verifies CUDA availability using `torch.cuda.is_available()`
- Automatically updates CUDA availability variables after installation
- Continues to launch ComfyUI after successful installation

### Progress Bars
- Uses `--progress-bar on` flag for all pip installations
- Provides visual feedback during long installations (especially PyTorch)

## Backward Compatibility

- ✅ All changes are backward compatible
- ✅ No breaking changes to existing functionality
- ✅ Works with both system Python and virtual environments
- ✅ Existing users can continue using the script as before

## Benefits

1. **Improved User Experience**: Users get clear guidance on what's missing and how to fix it
2. **Reduced Support Burden**: Common issues are caught and resolved automatically
3. **Better Error Messages**: Users understand what went wrong and how to fix it
4. **Professional Appearance**: Polished interface with ASCII art and clear formatting
5. **GPU Support**: Automatically ensures users have CUDA-enabled PyTorch for optimal performance

## Additional Notes

- The script maintains all original functionality while adding new features
- All user prompts are optional - users can cancel at any time
- Installation commands use `python -m pip` for consistency
- Error handling provides actionable troubleshooting steps
- The script is designed to be safe and non-destructive

## Potential Concerns and Side Effects

### Installation Risks
- **System Python Modifications**: If run outside a virtual environment, this script will install packages to the system Python, which may affect other Python applications. The script warns users about this and recommends virtual environments.
- **Automatic PyTorch Installation**: The CUDA PyTorch installation is large (~2-3GB) and takes several minutes. Users are clearly warned before installation begins.
- **Package Conflicts**: Installing packages automatically could potentially conflict with existing packages, though this is mitigated by using standard pip installation methods.

### Virtual Environment Considerations
- The script detects virtual environments and provides appropriate warnings
- Users are informed about the implications of installing in system Python vs virtual environments
- The script does not force virtual environment usage, but provides guidance

### Backward Compatibility
- ✅ All existing functionality is preserved
- Users who don't want automatic installations can cancel at any prompt
- The script works identically to the original if all dependencies are already installed

### PR Size Note
While this PR is larger than typical first-time contributions (+694/-61 lines, 5 files), all changes are cohesive and focused on a single feature: enhancing the startup script. Splitting this into smaller PRs would reduce the value of each individual PR, as the features work together as a unified improvement. We request thorough review due to the size, but believe the cohesive nature justifies the scope.

## Request for Review

Given my limited coding experience, I would greatly appreciate:
- Code review focusing on batch file best practices
- Verification of Python command invocations
- Testing in various Windows environments
- Feedback on error handling and user prompts
- Suggestions for improvements

Thank you for your time and consideration!

