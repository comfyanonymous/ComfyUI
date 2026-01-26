# ComfyUI Model Organizer

Bash script to organize and download ComfyUI models into proper directory structure.

## Usage

**Fresh Install (Download All Models):**
```bash
./Download_and_move.sh demo_model.txt /empty/dir /path/to/ComfyUI/models
```
Downloads all models from URLs in `demo_model.txt` directly to ComfyUI.

**Organize Existing Downloads:**
```bash
./Download_and_move.sh demo_model.txt /mnt/some/dir/pool/have/models /path/to/ComfyUI/models --move
```
Searches `/mnt/1T/Download` for matching files, moves them to proper subdirectories. Missing files are downloaded automatically.

Add `--dry-run` to preview actions without executing. Use `--copy` instead of `--move` to keep originals.
