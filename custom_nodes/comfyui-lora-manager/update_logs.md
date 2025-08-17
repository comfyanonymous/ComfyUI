# Update History

---

### v0.8.16
* **Dramatic Startup Speed Improvement** - Added cache serialization mechanism for significantly faster loading times, especially beneficial for large model collections
* **Enhanced Refresh Options** - Extended functionality with "Full Rebuild (complete)" option alongside "Quick Refresh (incremental)" to fix potential memory cache issues without requiring application restart
* **Customizable Display Density** - Replaced compact mode with adjustable display density settings for personalized layout customization
* **Model Creator Information** - Added creator details to model information panels for better attribution
* **Improved WebP Support** - Enhanced Save Image node with workflow embedding capability for WebP format images
* **Direct Example Access** - Added "Open Example Images Folder" button to card interfaces for convenient browsing of downloaded model examples
* **Enhanced Compatibility** - Full ComfyUI Desktop support for "Send lora or recipe to workflow" functionality
* **Cache Management** - Added settings to clear existing cache files when needed
* **Bug Fixes & Stability** - Various improvements for overall reliability and performance

### v0.8.15
* **Enhanced One-Click Integration** - Replaced copy button with direct send button allowing LoRAs/recipes to be sent directly to your current ComfyUI workflow without needing to paste
* **Flexible Workflow Integration** - Click to append LoRAs/recipes to existing loader nodes or Shift+click to replace content, with additional right-click menu options for "Send to Workflow (Append)" or "Send to Workflow (Replace)"
* **Improved LoRA Loader Controls** - Added header drag functionality for proportional strength adjustment of all LoRAs simultaneously (including CLIP strengths when expanded)
* **Keyboard Navigation Support** - Implemented Page Up/Down for page scrolling, Home key to jump to top, and End key to jump to bottom for faster browsing through large collections

### v0.8.14
* **Virtualized Scrolling** - Completely rebuilt rendering mechanism for smooth browsing with no lag or freezing, now supporting virtually unlimited model collections with optimized layouts for large displays, improving space utilization and user experience
* **Compact Display Mode** - Added space-efficient view option that displays more cards per row (7 on 1080p, 8 on 2K, 10 on 4K)
* **Enhanced LoRA Node Functionality** - Comprehensive improvements to LoRA loader/stacker nodes including real-time trigger word updates (reflecting any change anywhere in the LoRA chain for precise updates) and expanded context menu with "Copy Notes" and "Copy Trigger Words" options for faster workflow

### v0.8.13
* **Enhanced Recipe Management** - Added "Find duplicates" feature to identify and batch delete duplicate recipes with duplicate detection notifications during imports
* **Improved Source Tracking** - Source URLs are now saved with recipes imported via URL, allowing users to view original content with one click or manually edit links
* **Advanced LoRA Control** - Double-click LoRAs in Loader/Stacker nodes to access expanded CLIP strength controls for more precise adjustments of model and CLIP strength separately
* **Lycoris Model Support** - Added compatibility with Lycoris models for expanded creative options
* **Bug Fixes & UX Improvements** - Resolved various issues and enhanced overall user experience with numerous optimizations

### v0.8.12
* **Enhanced Model Discovery** - Added alphabetical navigation bar to LoRAs page for faster browsing through large collections
* **Optimized Example Images** - Improved download logic to automatically refresh stale metadata before fetching example images
* **Model Exclusion System** - New right-click option to exclude specific LoRAs or checkpoints from management
* **Improved Showcase Experience** - Enhanced interaction in LoRA and checkpoint showcase areas for better usability

### v0.8.11
* **Offline Image Support** - Added functionality to download and save all model example images locally, ensuring access even when offline or if images are removed from CivitAI or the site is down
* **Resilient Download System** - Implemented pause/resume capability with checkpoint recovery that persists through restarts or unexpected exits
* **Bug Fixes & Stability** - Resolved various issues to enhance overall reliability and performance

### v0.8.10
* **Standalone Mode** - Run LoRA Manager independently from ComfyUI for a lightweight experience that works even with other stable diffusion interfaces
* **Portable Edition** - New one-click portable version for easy startup and updates in standalone mode
* **Enhanced Metadata Collection** - Added support for SamplerCustomAdvanced node in the metadata collector module
* **Improved UI Organization** - Optimized Lora Loader node height to display up to 5 LoRAs at once with scrolling capability for larger collections

### v0.8.9
* **Favorites System** - New functionality to bookmark your favorite LoRAs and checkpoints for quick access and better organization
* **Enhanced UI Controls** - Increased model card button sizes for improved usability and easier interaction
* **Smoother Page Transitions** - Optimized interface switching between pages, eliminating flash issues particularly noticeable in dark theme
* **Bug Fixes & Stability** - Resolved various issues to enhance overall reliability and performance

### v0.8.8
* **Real-time TriggerWord Updates** - Enhanced TriggerWord Toggle node to instantly update when connected Lora Loader or Lora Stacker nodes change, without requiring workflow execution
* **Optimized Metadata Recovery** - Improved utilization of existing .civitai.info files for faster initialization and preservation of metadata from models deleted from CivitAI
* **Migration Acceleration** - Further speed improvements for users transitioning from A1111/Forge environments
* **Bug Fixes & Stability** - Resolved various issues to enhance overall reliability and performance

### v0.8.7
* **Enhanced Context Menu** - Added comprehensive context menu functionality to Recipes and Checkpoints pages for improved workflow
* **Interactive LoRA Strength Control** - Implemented drag functionality in LoRA Loader for intuitive strength adjustment
* **Metadata Collector Overhaul** - Rebuilt metadata collection system with optimized architecture for better performance
* **Improved Save Image Node** - Enhanced metadata capture and image saving performance with the new metadata collector
* **Streamlined Recipe Saving** - Optimized Save Recipe functionality to work independently without requiring Preview Image nodes
* **Bug Fixes & Stability** - Resolved various issues to enhance overall reliability and performance

### v0.8.6 Major Update
* **Checkpoint Management** - Added comprehensive management for model checkpoints including scanning, searching, filtering, and deletion
* **Enhanced Metadata Support** - New capabilities for retrieving and managing checkpoint metadata with improved operations
* **Improved Initial Loading** - Optimized cache initialization with visual progress indicators for better user experience

### v0.8.5
* **Enhanced LoRA & Recipe Connectivity** - Added Recipes tab in LoRA details to see all recipes using a specific LoRA
* **Improved Navigation** - New shortcuts to jump between related LoRAs and Recipes with one-click navigation
* **Video Preview Controls** - Added "Autoplay Videos on Hover" setting to optimize performance and reduce resource usage
* **UI Experience Refinements** - Smoother transitions between related content pages

### v0.8.4
* **Node Layout Improvements** - Fixed layout issues with LoRA Loader and Trigger Words Toggle nodes in newer ComfyUI frontend versions
* **Recipe LoRA Reconnection** - Added ability to reconnect deleted LoRAs in recipes by clicking the "deleted" badge in recipe details
* **Bug Fixes & Stability** - Resolved various issues for improved reliability

### v0.8.3
* **Enhanced Workflow Parser** - Rebuilt workflow analysis engine with improved support for ComfyUI core nodes and easier extensibility
* **Improved Recipe System** - Refined the experimental Save Recipe functionality with better workflow integration
* **New Save Image Node** - Added experimental node with metadata support for perfect CivitAI compatibility
  * Supports dynamic filename prefixes with variables [1](https://github.com/nkchocoai/ComfyUI-SaveImageWithMetaData?tab=readme-ov-file#filename_prefix)
* **Default LoRA Root Setting** - Added configuration option for setting your preferred LoRA directory

### v0.8.2  
* **Faster Initialization for Forge Users** - Improved first-run efficiency by utilizing existing `.json` and `.civitai.info` files from Forge’s CivitAI helper extension, making migration smoother.  
* **LoRA Filename Editing** - Added support for renaming LoRA files directly within LoRA Manager.  
* **Recipe Editing** - Users can now edit recipe names and tags.  
* **Retain Deleted LoRAs in Recipes** - Deleted LoRAs will remain listed in recipes, allowing future functionality to reconnect them once re-obtained.  
* **Download Missing LoRAs from Recipes** - Easily fetch missing LoRAs associated with a recipe.

### v0.8.1
* **Base Model Correction** - Added support for modifying base model associations to fix incorrect metadata for non-CivitAI LoRAs
* **LoRA Loader Flexibility** - Made CLIP input optional for model-only workflows like Hunyuan video generation
* **Expanded Recipe Support** - Added compatibility with 3 additional recipe metadata formats
* **Enhanced Showcase Images** - Generation parameters now displayed alongside LoRA preview images
* **UI Improvements & Bug Fixes** - Various interface refinements and stability enhancements

### v0.8.0
* **Introduced LoRA Recipes** - Create, import, save, and share your favorite LoRA combinations
* **Recipe Management System** - Easily browse, search, and organize your LoRA recipes
* **Workflow Integration** - Save recipes directly from your workflow with generation parameters preserved
* **Simplified Workflow Application** - Quickly apply saved recipes to new projects
* **Enhanced UI & UX** - Improved interface design and user experience
* **Bug Fixes & Stability** - Resolved various issues and enhanced overall performance

### v0.7.37
* Added NSFW content control settings (blur mature content and SFW-only filter)
* Implemented intelligent blur effects for previews and showcase media
* Added manual content rating option through context menu
* Enhanced user experience with configurable content visibility
* Fixed various bugs and improved stability

### v0.7.36
* Enhanced LoRA details view with model descriptions and tags display
* Added tag filtering system for improved model discovery
* Implemented editable trigger words functionality
* Improved TriggerWord Toggle node with new group mode option for granular control
* Added new Lora Stacker node with cross-compatibility support (works with efficiency nodes, ComfyRoll, easy-use, etc.)
* Fixed several bugs

### v0.7.35-beta
* Added base model filtering
* Implemented bulk operations (copy syntax, move multiple LoRAs)
* Added ability to edit LoRA model names in details view
* Added update checker with notification system
* Added support modal for user feedback and community links

### v0.7.33
* Enhanced LoRA Loader node with visual strength adjustment widgets
* Added toggle switches for LoRA enable/disable
* Implemented image tooltips for LoRA preview
* Added TriggerWord Toggle node with visual word selection
* Fixed various bugs and improved stability

### v0.7.3
* Added "Lora Loader (LoraManager)" custom node for workflows
* Implemented one-click LoRA integration
* Added direct copying of LoRA syntax from manager interface
* Added automatic preset strength value application
* Added automatic trigger word loading

### v0.7.0
* Added direct CivitAI integration for downloading LoRAs
* Implemented version selection for model downloads
* Added target folder selection for downloads
* Added context menu with quick actions
* Added force refresh for CivitAI data
* Implemented LoRA movement between folders
* Added personal usage tips and notes for LoRAs
* Improved performance for details window

## [Update 0.5.9] Enhanced Search Capabilities

- 🔍 **Advanced Search Features**:
  - Implemented fuzzy search for more flexible model finding
  - Added recursive search toggle functionality
  - Support for searching in current folder only or all subfolders

---

## [Update 0.5.8] UI Enhancements & Navigation Improvements

- ✨ **Enhanced Navigation**:
  - Added collapsible folder tags with persistent state
  - Implemented "Back to Top" button for easier browsing
  
- 🎨 **UI Refinements**: Various visual improvements and interface optimizations

---

## [Update 0.5.7] Performance Boost & Search Feature

- 🚀 **Major Performance Improvements**:
  - Implemented multi-layer caching and cache preloading
  - Added file system monitoring with incremental updates
  - Introduced pagination API with infinite scroll support
  
- 🔍 **Search Functionality**: New search feature to quickly find LoRA models
- 🐛 **Bug Fixes**: Various stability and performance improvements

---

## [Update 0.5.6] New Features and Optimizations

- 🛠️ **Code Refactor**: The codebase has been restructured to improve readability and maintainability, making it easier to manage and extend in future updates.

- 🚀 **Frontend Enhancements**: Significant performance improvements and refined user experience, including a more intuitive process for selecting and copying trigger words.

- 🔘 **New Menu Button**: A button has been added to the ComfyUI menu. Clicking it will open the LoRA Manager interface in a new window for quicker access.

---

## [Update 0.5.4] Support for Extra LoRA Paths via `extra_model_paths.yaml`

- 🛠️ **Extra LoRA Paths**: Additional flexibility has been introduced by supporting extra LoRA paths through the `extra_model_paths.yaml` file, allowing you to manage LoRAs from directories outside the default folder.

---

## [Update 0.5.3] Improved Preview Handling & Trigger Words Support

- ✅ **Smarter Preview Image Handling**: The manager now automatically scans for and uses existing local preview images. If a local preview is found, it will not re-download one from CivitAI when fetching model details, saving both time and bandwidth.

- 📝 **Trigger Words in LoRA Details**: Trigger words are now directly visible in the LoRA details window, making it easier to copy and integrate them into your workflows.

- ⚠️ **Note**: For automatic detection, ensure your local preview images are named using one of the following formats:
  - `<lora-file-name>.[png|jpg|jpeg|mp4]`
  - `<lora-file-name>.preview.[png|jpg|jpeg|mp4]`