---
name: writing-playwright-tests
description: 'Writes Playwright e2e tests for ComfyUI_frontend. Use when creating, modifying, or debugging browser tests. Triggers on: playwright, e2e test, browser test, spec file.'
---

# Writing Playwright Tests for ComfyUI_frontend

## Golden Rules

1. **ALWAYS look at existing tests first.** Search `browser_tests/tests/` for similar patterns before writing new tests.

2. **ALWAYS read the fixture code.** The APIs are in `browser_tests/fixtures/` - read them directly instead of guessing.

3. **Use premade JSON workflow assets** instead of building workflows programmatically.
   - Assets live in `browser_tests/assets/`
   - Load with `await comfyPage.workflow.loadWorkflow('feature/my_workflow')`
   - Create new assets by starting with `browser_tests/assets/default.json` and manually editing the JSON to match your desired graph state

## Vue Nodes vs LiteGraph: Decision Guide

Choose based on **what you're testing**, not personal preference:

| Testing...                                     | Use                              | Why                                      |
| ---------------------------------------------- | -------------------------------- | ---------------------------------------- |
| Vue-rendered node UI, DOM widgets, CSS states  | `comfyPage.vueNodes.*`           | Nodes are DOM elements, use locators     |
| Canvas interactions, connections, legacy nodes | `comfyPage.nodeOps.*`            | Canvas-based, use coordinates/references |
| Both in same test                              | Pick primary, minimize switching | Avoid confusion                          |

**Vue Nodes requires explicit opt-in:**

```typescript
await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
await comfyPage.vueNodes.waitForNodes()
```

**Vue Node state uses CSS classes:**

```typescript
const BYPASS_CLASS = /before:bg-bypass\/60/
await expect(node).toHaveClass(BYPASS_CLASS)
```

## Common Issues

These are frequent causes of flaky tests - check them first, but investigate if they don't apply:

| Symptom                            | Common Cause              | Typical Fix                                                                            |
| ---------------------------------- | ------------------------- | -------------------------------------------------------------------------------------- |
| Test passes locally, fails in CI   | Missing nextFrame()       | Add `await comfyPage.nextFrame()` after canvas ops (not needed after `loadWorkflow()`) |
| Keyboard shortcuts don't work      | Missing focus             | Add `await comfyPage.canvas.click()` first                                             |
| Double-click doesn't trigger       | Timing too fast           | Add `{ delay: 5 }` option                                                              |
| Elements end up in wrong position  | Drag animation incomplete | Use `{ steps: 10 }` not `{ steps: 1 }`                                                 |
| Widget value wrong after drag-drop | Upload incomplete         | Add `{ waitForUpload: true }`                                                          |
| Test fails when run with others    | Test pollution            | Add `afterEach` with `resetView()`                                                     |
| Local screenshots don't match CI   | Platform differences      | Screenshots are Linux-only, use PR label                                               |

## Test Tags

Add appropriate tags to every test:

| Tag           | When to Use                               |
| ------------- | ----------------------------------------- |
| `@smoke`      | Quick essential tests                     |
| `@slow`       | Tests > 10 seconds                        |
| `@screenshot` | Visual regression tests                   |
| `@canvas`     | Canvas interactions                       |
| `@node`       | Node-related                              |
| `@widget`     | Widget-related                            |
| `@mobile`     | Mobile viewport (runs on Pixel 5 project) |
| `@2x`         | HiDPI tests (runs on 2x scale project)    |

```typescript
test.describe('Feature', { tag: ['@screenshot', '@canvas'] }, () => {
```

## Retry Patterns

**Never use `waitForTimeout`** - it's always wrong.

| Pattern                  | Use Case                                             |
| ------------------------ | ---------------------------------------------------- |
| Auto-retrying assertions | `toBeVisible()`, `toHaveText()`, etc. (prefer these) |
| `expect.poll()`          | Single value polling                                 |
| `expect().toPass()`      | Multiple assertions that must all pass               |

```typescript
// Prefer auto-retrying assertions when possible
await expect(node).toBeVisible()

// Single value polling
await expect.poll(() => widget.getValue(), { timeout: 2000 }).toBe(100)

// Multiple conditions
await expect(async () => {
  expect(await node1.getValue()).toBe('foo')
  expect(await node2.getValue()).toBe('bar')
}).toPass({ timeout: 2000 })
```

## Screenshot Baselines

- **Screenshots are Linux-only.** Don't commit local screenshots.
- **To update baselines:** Add PR label `New Browser Test Expectations`
- **Mask dynamic content:**
  ```typescript
  await expect(comfyPage.canvas).toHaveScreenshot('page.png', {
    mask: [page.locator('.timestamp')]
  })
  ```

## CI Debugging

1. Download artifacts from failed CI run
2. Extract and view trace: `npx playwright show-trace trace.zip`
3. CI deploys HTML report to Cloudflare Pages (link in PR comment)
4. Reproduce CI: `CI=true pnpm test:browser`
5. Local runs: `pnpm test:browser:local`

## Anti-Patterns

Avoid these common mistakes:

1. **Arbitrary waits** - Use retrying assertions instead

   ```typescript
   // ❌ await page.waitForTimeout(500)
   // ✅ await expect(element).toBeVisible()
   ```

2. **Implementation-tied selectors** - Use test IDs or semantic selectors

   ```typescript
   // ❌ page.locator('div.container > button.btn-primary')
   // ✅ page.getByTestId('submit-button')
   ```

3. **Missing nextFrame after canvas ops** - Canvas needs sync time

   ```typescript
   await node.drag({ x: 50, y: 50 })
   await comfyPage.nextFrame() // Required
   ```

4. **Shared state between tests** - Tests must be independent
   ```typescript
   // ❌ let sharedData  // Outside test
   // ✅ Define state inside each test
   ```

## Quick Start Template

```typescript
// Path depends on test file location - adjust '../' segments accordingly
import {
  comfyPageFixture as test,
  comfyExpect as expect
} from '../fixtures/ComfyPage'

test.describe('FeatureName', { tag: ['@canvas'] }, () => {
  test.afterEach(async ({ comfyPage }) => {
    await comfyPage.canvasOps.resetView()
  })

  test('should do something', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('myWorkflow')

    const node = (await comfyPage.nodeOps.getNodeRefsByTitle('KSampler'))[0]
    // ... test logic

    await expect(comfyPage.canvas).toHaveScreenshot('expected.png')
  })
})
```

## Finding Patterns

```bash
# Find similar tests
grep -r "KSampler" browser_tests/tests/

# Find usage of a fixture method
grep -r "loadWorkflow" browser_tests/tests/

# Find tests with specific tag
grep -r '@screenshot' browser_tests/tests/
```

## Key Files to Read

| Purpose           | Path                                       |
| ----------------- | ------------------------------------------ |
| Main fixture      | `browser_tests/fixtures/ComfyPage.ts`      |
| Helper classes    | `browser_tests/fixtures/helpers/`          |
| Component objects | `browser_tests/fixtures/components/`       |
| Test selectors    | `browser_tests/fixtures/selectors.ts`      |
| Vue Node helpers  | `browser_tests/fixtures/VueNodeHelpers.ts` |
| Test assets       | `browser_tests/assets/`                    |
| Existing tests    | `browser_tests/tests/`                     |

**Read the fixture code directly** - it's the source of truth for available methods.
