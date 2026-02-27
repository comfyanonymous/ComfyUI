import { expect } from '@playwright/test'

import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

/**
 * Tests that templates are automatically fitted to view when loaded.
 *
 * When openSource === 'template', fitView() is called to ensure
 * templates with saved off-screen viewport positions (extra.ds)
 * are always displayed correctly.
 */
test.describe('Template Fit View', { tag: ['@canvas', '@workflow'] }, () => {
  test('should automatically fit view when loading a template with off-screen saved position', async ({
    comfyPage
  }) => {
    await comfyPage.settings.setSetting('Comfy.EnableWorkflowViewRestore', true)

    // Serialize the current default graph, inject an extreme off-screen
    // viewport position, then reload it as a template. Without the fix,
    // the saved offset [-5000, -5000] would be restored and nodes would
    // be invisible.
    const viewportState = await comfyPage.page.evaluate(async () => {
      const app = window.app!
      const workflow = app.graph.serialize()

      workflow.extra = {
        ...workflow.extra,
        ds: { scale: 1, offset: [-5000, -5000] }
      }

      await app.loadGraphData(workflow as ComfyWorkflowJSON, true, true, null, {
        openSource: 'template'
      })

      return {
        offsetX: app.canvas.ds.offset[0],
        offsetY: app.canvas.ds.offset[1],
        nodeCount: app.graph._nodes.length
      }
    })

    expect(viewportState.nodeCount).toBeGreaterThan(0)

    // fitView() should have overridden the saved [-5000, -5000] offset
    expect(
      viewportState.offsetX,
      'Viewport X offset should not be the saved off-screen value'
    ).not.toBe(-5000)
    expect(
      viewportState.offsetY,
      'Viewport Y offset should not be the saved off-screen value'
    ).not.toBe(-5000)
  })
})
