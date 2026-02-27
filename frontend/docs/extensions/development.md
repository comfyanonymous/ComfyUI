# Extension Development Guide

## Understanding Extensions in ComfyUI

### Terminology Clarification

**ComfyUI Extension** - The umbrella term for any 3rd party code that extends ComfyUI functionality. This includes:

1. **Python Custom Nodes** - Backend nodes providing new operations
   - Located in `/custom_nodes/` directories
   - Registered via Python module system
   - Identified by `custom_nodes.<name>` in `python_module` field

2. **JavaScript Extensions** - Frontend functionality that can be:
   - Pure JavaScript extensions (implement `ComfyExtension` interface)
   - JavaScript components of custom nodes (in `/web/` or `/js/` folders, or custom directories specified via `WEB_DIRECTORY` export in `__init__.py` [see docs](https://docs.comfy.org/custom-nodes/backend/lifecycle#web-directory))
   - Core extensions (built into frontend at `/src/extensions/core/` - see [Core Extensions Documentation](./core.md))

### How Extensions Load

**Backend Flow (Python Custom Nodes):**

1. ComfyUI server starts → scans `/custom_nodes/` directories
2. Loads Python modules (e.g., `/custom_nodes/ComfyUI-Impact-Pack/__init__.py`)
3. Python code registers new node types with the server
4. Server exposes these via `/object_info` API with metadata like `python_module: "custom_nodes.ComfyUI-Impact-Pack"`
5. These nodes execute on the server when workflows run

**Frontend Flow (JavaScript):**

_Core Extensions (always available):_

1. Built directly into the frontend bundle at `/src/extensions/core/`
2. Loaded immediately when the frontend starts
3. No network requests needed - they're part of the compiled code

_Custom Node JavaScript (loaded dynamically):_

1. Frontend starts → calls `/extensions` API
2. Server responds with list of JavaScript files from:
   - `/web/extensions/*.js` (legacy location)
   - `/custom_nodes/*/web/*.js` (node-specific UI code)
3. Frontend fetches each JavaScript file (e.g., `/extensions/ComfyUI-Impact-Pack/impact.js`)
4. JavaScript executes immediately, calling `app.registerExtension()` to hook into the UI
5. These registered hooks enhance the UI for their associated Python nodes

**The Key Distinction:**

- **Python nodes** = Backend processing (what shows in your node menu)
- **JavaScript extensions** = Frontend enhancements (how nodes look/behave in the UI)
- A custom node package can have both, just Python, or (rarely) just JavaScript

## Why Extensions Don't Load in Dev Server

The development server cannot load custom node JavaScript due to architectural constraints from the TypeScript/Vite migration.

### The Technical Challenge

ComfyUI migrated to TypeScript and Vite, but thousands of extensions rely on the old unbundled module system. The solution was a **shim system** that maintains backward compatibility - but only in production builds.

### How the Shim Works

**Production Build:**
During production build, a custom Vite plugin:

- Binds all module exports to `window.comfyAPI`
- Generates shim files that re-export from this global object

```javascript
// Original source: /scripts/api.ts
export const api = { }

// Generated shim: /scripts/api.js
export * from window.comfyAPI.modules['/scripts/api.js']

// Extension imports work unchanged:
import { api } from '/scripts/api.js'
```

**Why Dev Server Can't Support This:**

- The dev server serves raw source files without bundling
- Vite refuses to transform node_modules in unbundled mode
- Creating real-time shims would require intercepting every module request
- This would defeat the purpose of a fast dev server

### The Trade-off

This was the least friction approach:

- ✅ Extensions work in production without changes
- ✅ Developers get modern tooling (TypeScript, hot reload)
- ❌ Extension testing requires production build or workarounds

The alternative would have been breaking all existing extensions or staying with the legacy build system.

## Development Workarounds

### Option 1: Develop as Core Extension (Recommended)

1. Copy your extension to `src/extensions/core/` (see [Core Extensions Documentation](./core.md) for existing core extensions and architecture)
2. Update imports to relative paths:
   ```javascript
   import { app } from '../../scripts/app'
   import { api } from '../../scripts/api'
   ```
3. Add to `src/extensions/core/index.ts`
4. Test with hot reload working
5. Move back when complete

### Option 2: Use Production Build

Build the frontend for full functionality:

```bash
pnpm build
```

For faster iteration during development, use watch mode:

```bash
pnpm exec vite build --watch
```

Note: Watch mode provides faster rebuilds than full builds, but still no hot reload

### Option 3: Test Against Cloud/Staging

For cloud extensions, modify `.env`:

```
DEV_SERVER_COMFYUI_URL=http://stagingcloud.comfy.org/
```

## Key Points

- Python nodes work normally in dev mode
- JavaScript extensions require workarounds or production builds
- Core extensions provide built-in functionality - see [Core Extensions Documentation](./core.md) for the complete list
- The `ComfyExtension` interface defines all available hooks for extending the frontend

## Further Reading

- [Core Extensions Architecture](./core.md) - Complete list of core extensions and development guidelines
- [JavaScript Extension Hooks](https://docs.comfy.org/custom-nodes/js/javascript_hooks) - Official documentation on extension hooks
- [ComfyExtension Interface](../../src/types/comfy.ts) - TypeScript interface defining all extension capabilities
