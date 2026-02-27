# Playwright Testing for ComfyUI_frontend

This document outlines the setup, usage, and common patterns for Playwright browser tests in the ComfyUI_frontend project.

## Prerequisites

**CRITICAL**: Start ComfyUI backend with `--multi-user` flag:

```bash
python main.py --multi-user
```

Without this flag, parallel tests will conflict and fail randomly.

## Setup

### ComfyUI devtools

ComfyUI_devtools is included in this repository under `tools/devtools/`. During CI/CD, these files are automatically copied to the `custom_nodes` directory.  
ComfyUI_devtools adds additional API endpoints and nodes to ComfyUI for browser testing.

For local development, copy the devtools files to your ComfyUI installation:

```bash
cp -r tools/devtools/* /path/to/your/ComfyUI/custom_nodes/ComfyUI_devtools/
```

### Node.js & Playwright Prerequisites

Ensure you have Node.js v20 or v22 installed. Then, set up the Chromium test driver:

```bash
pnpm exec playwright install chromium --with-deps
```

### Environment Configuration

Create `.env` from the template:

```bash
cp .env_example .env
```

Key settings for debugging:

```bash
# Remove Vue dev overlay that blocks UI elements
DISABLE_VUE_PLUGINS=true

# Test against dev server (recommended) or backend directly
PLAYWRIGHT_TEST_URL=http://localhost:5173  # Dev server
# PLAYWRIGHT_TEST_URL=http://localhost:8188  # Direct backend

# Path to ComfyUI for backing up user data/settings before tests
TEST_COMFYUI_DIR=/path/to/your/ComfyUI
```

### Common Setup Issues

### Release API Mocking

By default, all tests mock the release API (`api.comfy.org/releases`) to prevent release notification popups from interfering with test execution. This is necessary because the release notifications can appear over UI elements and block test interactions.

To test with real release data, you can disable mocking:

```typescript
await comfyPage.setup({ mockReleases: false })
```

For tests that specifically need to test release functionality, see the example in `tests/releaseNotifications.spec.ts`.

## Running Tests

**Always use UI mode for development:**

```bash
pnpm exec playwright test --ui
```

UI mode features:

- **Locator picker**: Click the target icon, then click any element to get the exact locator code to use in your test. The code appears in the _Locator_ tab.
- **Step debugging**: Step through your test line-by-line by clicking _Source_ tab
- **Time travel**: In the _Actions_ tab/panel, click any step to see the browser state at that moment
- **Console/Network Tabs**: View logs and API calls at each step
- **Attachments Tab**: View all snapshots with expected and actual images

![Playwright UI Mode](https://github.com/user-attachments/assets/9b9cb09f-6da7-4fa0-81df-2effceced755)

For CI or headless testing:

```bash
pnpm exec playwright test                    # Run all tests
pnpm exec playwright test widget.spec.ts     # Run specific test file
```

### Local Development Config

For debugging, you can try adjusting these settings in `playwright.config.ts`:

```typescript
export default defineConfig({
  // VERY HELPFUL: Skip screenshot tests locally
  grep: process.env.CI ? undefined : /^(?!.*screenshot).*$/

  retries: 0, // No retries while debugging. Increase if writing new tests. that may be flaky.
  workers: 1, // Single worker for easier debugging. Increase to match CPU cores if you want to run a lot of tests in parallel.
  timeout: 30000, // Longer timeout for breakpoints

  use: {
    trace: 'on', // Always capture traces (CI uses 'on-first-retry')
    video: 'on' // Always record video (CI uses 'retain-on-failure')
  },

})
```

## Test Structure

Browser tests in this project follow a specific organization pattern:

- **Fixtures**: Located in `fixtures/` - These provide test setup and utilities
  - `ComfyPage.ts` - The main fixture for interacting with ComfyUI
  - `ComfyMouse.ts` - Utility for mouse interactions with the canvas
  - Components fixtures in `fixtures/components/` - Page object models for UI components

- **Tests**: Located in `tests/` - The actual test specifications
  - Organized by functionality (e.g., `widget.spec.ts`, `interaction.spec.ts`)
  - Snapshot directories (e.g., `widget.spec.ts-snapshots/`) contain reference screenshots

- **Utilities**: Located in `utils/` - Common utility functions
  - `litegraphUtils.ts` - Utilities for working with LiteGraph nodes

## Writing Effective Tests

When writing new tests, follow these patterns:

### Test Structure

```typescript
// Import the test fixture
import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Feature Name', () => {
  // Set up test environment if needed
  test.beforeEach(async ({ comfyPage }) => {
    // Common setup
  })

  test('should do something specific', async ({ comfyPage }) => {
    // Test implementation
  })
})
```

### Leverage Existing Fixtures and Helpers

Always check for existing helpers and fixtures before implementing new ones:

- **ComfyPage**: Main fixture with methods for canvas interaction and node management
- **ComfyMouse**: Helper for precise mouse operations on the canvas
- **Helpers**: Check `browser_tests/helpers/` for specialized helpers like:
  - `actionbar.ts`: Interact with the action bar
  - `manageGroupNode.ts`: Group node management operations
  - `templates.ts`: Template workflows operations
- **Component Fixtures**: Check `browser_tests/fixtures/components/` for UI component helpers
- **Utility Functions**: Check `browser_tests/utils/` and `browser_tests/fixtures/utils/` for shared utilities

Most common testing needs are already addressed by these helpers, which will make your tests more consistent and reliable.

### Key Testing Patterns

1. **Focus elements explicitly**:
   Canvas-based elements often need explicit focus before interaction:

   ```typescript
   // Click the canvas first to focus it before pressing keys
   await comfyPage.canvas.click()
   await comfyPage.page.keyboard.press('a')
   ```

2. **Mark canvas as dirty if needed**:
   Some interactions need explicit canvas updates:

   ```typescript
   // After programmatically changing node state, mark canvas dirty
   await comfyPage.page.evaluate(() => {
     window['app'].graph.setDirtyCanvas(true, true)
   })
   ```

3. **Use node references over coordinates**:
   Node references from `fixtures/utils/litegraphUtils.ts` provide stable ways to interact with nodes:

   ```typescript
   // Prefer this:
   const node = await comfyPage.getNodeRefsByType('LoadImage')[0]
   await node.click('title')

   // Over this:
   await comfyPage.canvas.click({ position: { x: 100, y: 100 } })
   ```

4. **Wait for canvas to render after UI interactions**:

   ```typescript
   await comfyPage.nextFrame()
   ```

5. **Clean up persistent server state**:
   While most state is reset between tests, anything stored on the server persists:

   ```typescript
   // Reset settings that affect other tests (these are stored on server)
   await comfyPage.setSetting('Comfy.ColorPalette', 'dark')
   await comfyPage.setSetting('Comfy.NodeBadge.NodeIdBadgeMode', 'None')

   // Clean up uploaded files if needed
   await comfyPage.request.delete(`${comfyPage.url}/api/delete/image.png`)
   ```

6. **Prefer functional assertions over screenshots**:
   Use screenshots only when visual verification is necessary:

   ```typescript
   // Prefer this:
   expect(await node.isPinned()).toBe(true)
   expect(await node.getProperty('title')).toBe('Expected Title')

   // Over this - only use when needed:
   await expect(comfyPage.canvas).toHaveScreenshot('state.png')
   ```

7. **Use minimal test workflows**:
   When creating test workflows, keep them as minimal as possible:

   ```typescript
   // Include only the components needed for the test
   await comfyPage.loadWorkflow('single_ksampler')
   ```

8. **Debug helpers for visual debugging** (remove before committing):

   ComfyPage includes temporary debug methods for troubleshooting:

   ```typescript
   test('debug failing interaction', async ({ comfyPage }, testInfo) => {
     // Add visual markers to see click positions
     await comfyPage.debugAddMarker({ x: 100, y: 200 })

     // Attach screenshot with markers to test report
     await comfyPage.debugAttachScreenshot(testInfo, 'node-positions', {
       element: 'canvas',
       markers: [{ position: { x: 100, y: 200 } }]
     })

     // Show canvas overlay for easier debugging
     await comfyPage.debugShowCanvasOverlay()

     // Remember to remove debug code before committing!
   })
   ```

   Available debug methods:
   - `debugAddMarker(position)` - Red circle at position
   - `debugAttachScreenshot(testInfo, name)` - Attach to test report
   - `debugShowCanvasOverlay()` - Show canvas as overlay
   - `debugGetCanvasDataURL()` - Get canvas as base64

## Common Patterns and Utilities

### Page Object Pattern

Tests use the Page Object pattern to create abstractions over the UI:

```typescript
// Using the ComfyPage fixture
test('Can toggle boolean widget', async ({ comfyPage }) => {
  await comfyPage.loadWorkflow('widgets/boolean_widget')
  const node = (await comfyPage.getFirstNodeRef())!
  const widget = await node.getWidget(0)
  await widget.click()
})
```

### Node References

The `NodeReference` class provides helpers for interacting with LiteGraph nodes:

```typescript
// Getting node by type and interacting with it
const nodes = await comfyPage.getNodeRefsByType('LoadImage')
const loadImageNode = nodes[0]
const widget = await loadImageNode.getWidget(0)
await widget.click()
```

### Visual Regression Testing

Tests use screenshot comparisons to verify UI state:

```typescript
// Take a screenshot and compare to reference
await expect(comfyPage.canvas).toHaveScreenshot('boolean_widget_toggled.png')
```

### Waiting for Animations

Always call `nextFrame()` after actions that trigger animations:

```typescript
await comfyPage.canvas.click({ position: { x: 100, y: 100 } })
await comfyPage.nextFrame() // Wait for canvas to redraw
```

### Mouse Interactions

Canvas operations use special helpers to ensure proper timing:

```typescript
// Using ComfyMouse for drag and drop
await comfyMouse.dragAndDrop(
  { x: 100, y: 100 }, // From
  { x: 200, y: 200 } // To
)

// Standard ComfyPage helpers
await comfyPage.drag({ x: 100, y: 100 }, { x: 200, y: 200 })
await comfyPage.pan({ x: 200, y: 200 })
await comfyPage.zoom(-100) // Zoom in
```

### Workflow Management

Tests use workflows stored in `assets/` for consistent starting points:

```typescript
// Load a test workflow
await comfyPage.loadWorkflow('single_ksampler')

// Wait for workflow to load and stabilize
await comfyPage.nextFrame()
```

### Custom Assertions

The project includes custom Playwright assertions through `comfyExpect`:

```typescript
// Check if a node is in a specific state
await expect(node).toBePinned()
await expect(node).toBeBypassed()
await expect(node).toBeCollapsed()
```

## Troubleshooting Common Issues

### Flaky Tests

- **Timing Issues**: Always wait for animations to complete with `nextFrame()`
- **Coordinate Sensitivity**: Canvas coordinates are viewport-relative; use node references when possible
- **Test Isolation**: Tests run in parallel; avoid dependencies between tests
- **Screenshots vary**: Ensure your OS and browser match the reference environment (Linux)
- **Async / await**: Race conditions are a very common cause of test flakiness

## Screenshot Testing

Due to variations in system font rendering, screenshot expectations are platform-specific. Please note:

- **Do not commit local screenshot expectations** to the repository
- We maintain Linux screenshot expectations as our GitHub Action runner operates in a Linux environment
- While developing, you can generate local screenshots for your tests, but these will differ from CI-generated ones

### Working with Screenshots Locally

Option 1 - Skip screenshot tests (add to `playwright.config.ts`):

```typescript
export default defineConfig({
  grep: process.env.CI ? undefined : /^(?!.*screenshot).*$/
})
```

Option 2 - Generate local baselines for comparison:

```bash
pnpm exec playwright test --update-snapshots
```

### Creating New Screenshot Baselines

For PRs from `Comfy-Org/ComfyUI_frontend` branches:

1. Write test with `toHaveScreenshot('filename.png')`
2. Create PR and add `New Browser Test Expectation` label
3. CI will generate and commit the Linux baseline screenshots

> **Note:** Fork PRs cannot auto-commit screenshots. A maintainer will need to commit the screenshots manually for you (don't worry, they'll do it).

## Viewing Test Reports

### Automated Test Deployment

The project automatically deploys Playwright test reports to Cloudflare Pages for every PR and push to main branches.

### Accessing Test Reports

- **From PR comments**: Click the "View Report" links for each browser
- **Direct URLs**: Reports are available at `https://[branch].comfyui-playwright-[browser].pages.dev` (branch-specific deployments)
- **From GitHub Actions**: Download artifacts from failed runs

### How It Works

1. **Test execution**: All browser tests run in parallel across multiple browsers
2. **Report generation**: HTML reports are generated for each browser configuration
3. **Cloudflare deployment**: Each browser's report deploys to its own Cloudflare Pages project with branch isolation:
   - `comfyui-playwright-chromium` (with branch-specific URLs)
   - `comfyui-playwright-mobile-chrome` (with branch-specific URLs)
   - `comfyui-playwright-chromium-2x` (2x scale, with branch-specific URLs)
   - `comfyui-playwright-chromium-0-5x` (0.5x scale, with branch-specific URLs)

4. **PR comments**: GitHub automatically updates PR comments with:
   - ✅/❌ Test status for each browser
   - Direct links to interactive test reports
   - Real-time progress updates as tests complete

## Resources

- [Playwright UI Mode](https://playwright.dev/docs/test-ui-mode) - Interactive test debugging
- [Playwright Debugging Guide](https://playwright.dev/docs/debug)
- [act](https://github.com/nektos/act) - Run GitHub Actions locally for CI debugging
