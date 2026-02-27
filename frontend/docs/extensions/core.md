# Core Extensions

This directory contains the core extensions that provide essential functionality to the ComfyUI frontend.

## Table of Contents

- [Overview](#overview)
- [Extension Architecture](#extension-architecture)
- [Core Extensions](#core-extensions)
- [Extension Development](#extension-development)
- [Extension Hooks](#extension-hooks)
- [Further Reading](#further-reading)

## Overview

Extensions in ComfyUI are modular JavaScript modules that extend and enhance the functionality of the frontend. The extensions in this directory are considered "core" as they provide fundamental features that are built into ComfyUI by default.

## Extension Architecture

ComfyUI's extension system follows these key principles:

1. **Registration-based:** Extensions must register themselves with the application using `app.registerExtension()`
2. **Hook-driven:** Extensions interact with the system through predefined hooks
3. **Non-intrusive:** Extensions should avoid directly modifying core objects where possible

## Core Extensions List

The following table lists ALL core extensions in the system as of 2025-01-30:

### Main Extensions

| Extension               | Description                                                   | Category  |
| ----------------------- | ------------------------------------------------------------- | --------- |
| clipspace.ts            | Implements the Clipspace feature for temporary image storage  | Image     |
| contextMenuFilter.ts    | Provides context menu filtering capabilities                  | UI        |
| dynamicPrompts.ts       | Provides dynamic prompt generation capabilities               | Prompts   |
| editAttention.ts        | Implements attention editing functionality                    | Text      |
| electronAdapter.ts      | Adapts functionality for Electron environment                 | Platform  |
| groupNode.ts            | Implements the group node functionality to organize workflows | Graph     |
| groupNodeManage.ts      | Provides group node management operations                     | Graph     |
| groupOptions.ts         | Handles group node configuration options                      | Graph     |
| index.ts                | Main extension registration and coordination                  | Core      |
| load3d.ts               | Supports 3D model loading and visualization                   | 3D        |
| maskeditor.ts           | Implements the mask editor for image masking operations       | Image     |
| nodeTemplates.ts        | Provides node template functionality                          | Templates |
| noteNode.ts             | Adds note nodes for documentation within workflows            | Graph     |
| previewAny.ts           | Universal preview functionality for various data types        | Preview   |
| rerouteNode.ts          | Implements reroute nodes for cleaner workflow connections     | Graph     |
| saveImageExtraOutput.ts | Handles additional image output saving                        | Image     |
| saveMesh.ts             | Implements 3D mesh saving functionality                       | 3D        |
| simpleTouchSupport.ts   | Provides basic touch interaction support                      | Input     |
| slotDefaults.ts         | Manages default values for node slots                         | Nodes     |
| uploadAudio.ts          | Handles audio file upload functionality                       | Audio     |
| uploadImage.ts          | Handles image upload functionality                            | Image     |
| webcamCapture.ts        | Provides webcam capture capabilities                          | Media     |
| widgetInputs.ts         | Implements various widget input types                         | Widgets   |

### Conditional Lines Subdirectory

Located in `extensions/core/load3d/conditional-lines/`:

| File                        | Description                             |
| --------------------------- | --------------------------------------- |
| ColoredShadowMaterial.js    | Material for colored shadow rendering   |
| ConditionalEdgesGeometry.js | Geometry for conditional edge rendering |
| ConditionalEdgesShader.js   | Shader for conditional edges            |
| OutsideEdgesGeometry.js     | Geometry for outside edge detection     |

### Lines2 Subdirectory

Located in `extensions/core/load3d/conditional-lines/Lines2/`:

| File                               | Description                             |
| ---------------------------------- | --------------------------------------- |
| ConditionalLineMaterial.js         | Material for conditional line rendering |
| ConditionalLineSegmentsGeometry.js | Geometry for conditional line segments  |

### ThreeJS Override Subdirectory

Located in `extensions/core/load3d/threejsOverride/`:

| File                 | Description                                   |
| -------------------- | --------------------------------------------- |
| OverrideMTLLoader.js | Custom MTL loader with enhanced functionality |

## Extension Development

When developing or modifying extensions, follow these best practices:

1. **Use provided hooks** rather than directly modifying core application objects
2. **Maintain compatibility** with other extensions
3. **Follow naming conventions** for both extension names and settings
4. **Properly document** extension hooks and functionality
5. **Test with other extensions** to ensure no conflicts

### Extension Registration

Extensions are registered using the `app.registerExtension()` method:

```javascript
app.registerExtension({
  name: 'MyExtension',

  // Hook implementations
  async init() {
    // Implementation
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // Implementation
  }

  // Other hooks as needed
})
```

## Extension Hooks

ComfyUI extensions can implement various hooks that are called at specific points in the application lifecycle:

### Hook Execution Sequence

#### Web Page Load

```
init
addCustomNodeDefs
getCustomWidgets
beforeRegisterNodeDef    [repeated multiple times]
registerCustomNodes
beforeConfigureGraph
nodeCreated
loadedGraphNode
afterConfigureGraph
setup
```

#### Loading Workflow

```
beforeConfigureGraph
beforeRegisterNodeDef   [zero, one, or multiple times]
nodeCreated             [repeated multiple times]
loadedGraphNode         [repeated multiple times]
afterConfigureGraph
```

#### Adding New Node

```
nodeCreated
```

### Key Hooks

| Hook                          | Description                                                |
| ----------------------------- | ---------------------------------------------------------- |
| `init`                        | Called after canvas creation but before nodes are added    |
| `setup`                       | Called after the application is fully set up and running   |
| `addCustomNodeDefs`           | Called before nodes are registered with the graph          |
| `getCustomWidgets`            | Allows extensions to add custom widgets                    |
| `beforeRegisterNodeDef`       | Allows extensions to modify nodes before registration      |
| `registerCustomNodes`         | Allows extensions to register additional nodes             |
| `loadedGraphNode`             | Called when a node is reloaded onto the graph              |
| `nodeCreated`                 | Called after a node's constructor                          |
| `beforeConfigureGraph`        | Called before a graph is configured                        |
| `afterConfigureGraph`         | Called after a graph is configured                         |
| `getSelectionToolboxCommands` | Allows extensions to add commands to the selection toolbox |

For the complete list of available hooks and detailed descriptions, see the [ComfyExtension interface in comfy.ts](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/types/comfy.ts).

## Further Reading

For more detailed information about ComfyUI's extension system, refer to the official documentation:

- [JavaScript Extension Overview](https://docs.comfy.org/custom-nodes/js/javascript_overview)
- [JavaScript Hooks](https://docs.comfy.org/custom-nodes/js/javascript_hooks)
- [JavaScript Objects and Hijacking](https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking)
- [JavaScript Settings](https://docs.comfy.org/custom-nodes/js/javascript_settings)
- [JavaScript Examples](https://docs.comfy.org/custom-nodes/js/javascript_examples)

Also, check the main [README.md](https://github.com/Comfy-Org/ComfyUI_frontend#developer-apis) section on Developer APIs for the latest information on extension APIs and features.
