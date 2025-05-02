# User Guide

This document provides detailed information on how to use ComfyUI effectively.

## Table of Contents
- [Getting Started](#getting-started)
- [Interface Overview](#interface-overview)
- [Working with Nodes](#working-with-nodes)
- [Managing Models](#managing-models)
- [Creating Workflows](#creating-workflows)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)

## Getting Started

After installing ComfyUI, you can access it via your web browser:

1. Start ComfyUI:
   ```bash
   comfyui
   # OR if installed from source
   python main.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8188
   ```

3. ComfyUI will load with an empty workspace where you can create your nodes and workflow.

## Interface Overview

### Main Areas

- **Canvas**: The main area where you create and connect nodes
- **Node Menu**: Left sidebar with available nodes categorized by function
- **Settings**: Access global settings via the gear icon in the top right
- **Queue**: View and manage processing queue in the top panel

### Navigation

- **Pan**: Middle mouse button or Alt + Left mouse button
- **Zoom**: Mouse wheel or Ctrl + Mouse wheel
- **Select multiple nodes**: Drag a selection box or Shift + Click
- **Move nodes**: Drag selected nodes
- **Delete nodes**: Select and press Delete key

## Working with Nodes

### Node Basics

Nodes are the building blocks of ComfyUI workflows. Each node performs a specific function and can be connected to other nodes.

#### Node Types

- **Input Nodes**: Provide data (text prompts, images, etc.)
- **Processing Nodes**: Transform data (models, samplers, conditioning)
- **Output Nodes**: Generate results (images, videos, latents)
- **Utility Nodes**: Helper functions (math, logic, conversion)

#### Creating Nodes

1. Right-click on the canvas
2. Navigate through the menu categories
3. Click on the node you want to add
4. Alternatively, use the Quick Node Search with Ctrl+Space

#### Connecting Nodes

1. Click and drag from an output socket (right side)
2. Connect to an input socket (left side)
3. Compatible connections will be highlighted
4. Incompatible connections will be rejected

### Node Properties

Each node has properties that can be configured:

1. Click on a node to select it
2. Adjust parameters in the property panel
3. Some parameters support dynamic values via connections

## Managing Models

ComfyUI supports various AI models for different purposes.

### Model Types

- **Checkpoints**: Main Stable Diffusion models
- **LoRA**: Style adaptations and fine-tuning
- **Textual Inversions/Embeddings**: Concept encodings
- **VAE**: Variational autoencoders for encoding/decoding
- **CLIP**: Text encoders for prompts
- **ControlNet**: Models for guided image generation

### Loading Models

Models can be loaded via dedicated nodes:

1. Add the appropriate model loader node
2. Select your model from the dropdown
3. Connect to other nodes that require the model

### Model Management

- Models are stored in the `models` directory
- Subdirectories organize different model types
- Add new models by placing them in the appropriate folder
- ComfyUI automatically detects new models on startup

## Creating Workflows

### Basic Workflow

A minimal image generation workflow consists of:

1. **Checkpoint Loader**: Load a model
2. **CLIP Text Encode**: Process positive and negative prompts
3. **KSampler**: Configure sampling parameters
4. **VAE Decode**: Convert latent representation to image
5. **Save Image**: Output the generated image

### Workflow Management

- **Save Workflow**: Click "Save" to download JSON workflow file
- **Load Workflow**: Click "Load" to import a workflow
- **Share Workflows**: Exchange JSON files with others
- **API Access**: Workflows can be executed via API

### Workflow Templates

ComfyUI includes templates for common tasks:

1. Click "Load" to access templates
2. Select a template that matches your need
3. Modify parameters to customize the results

## Advanced Features

### Batching

Process multiple images with one workflow:

1. Use batch size parameter in sampler nodes
2. Use Empty Latent Image with multiple batches
3. Process results with Latent from Batch node

### Animation

Create animations using keyframes or video input:

1. Use Animate Diff or similar animation nodes
2. Configure frames, motion, and interpolation
3. Output video or image sequence

### Upscaling

Enhance resolution of generated images:

1. Generate base image
2. Feed into upscaler node (Lanczos, ESRGAN, etc.)
3. Configure scale factor and parameters
4. Save higher resolution result

## Best Practices

### Performance Optimization

- Use half-precision (fp16) for faster processing
- Adjust VRAM usage in settings
- Process at lower resolution and upscale later
- Use VAE tiling for large images

### Workflow Organization

- Group related nodes for clarity
- Add comment nodes to document workflows
- Use consistent naming conventions
- Break complex workflows into subgraphs

### Common Pitfalls

- Ensure compatible node connections
- Watch VRAM usage for larger models
- Backup workflows regularly
- Check error messages in console log

### Getting Help

- Hover over node inputs/outputs for tooltips
- Check documentation for specific nodes
- Visit Discord community for support
- Explore shared workflows for examples