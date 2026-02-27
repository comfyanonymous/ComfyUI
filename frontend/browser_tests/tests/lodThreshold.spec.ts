import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('LOD Threshold', { tag: ['@screenshot', '@canvas'] }, () => {
  test('Should switch to low quality mode at correct zoom threshold', async ({
    comfyPage
  }) => {
    // Load a workflow with some nodes to render
    await comfyPage.workflow.loadWorkflow('default')

    // Get initial LOD state and settings
    const initialState = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        lowQuality: canvas.low_quality,
        scale: canvas.ds.scale,
        minFontSize: canvas.min_font_size_for_lod
      }
    })

    // Should start at normal zoom (not low quality)
    expect(initialState.lowQuality).toBe(false)
    expect(initialState.scale).toBeCloseTo(1, 1)

    // Calculate expected threshold (8px / 14px ≈ 0.571)
    const expectedThreshold = initialState.minFontSize / 14
    // Can't access private _lowQualityZoomThreshold directly

    // Zoom out just above threshold (should still be high quality)
    await comfyPage.canvasOps.zoom(120, 5) // Zoom out 5 steps
    await comfyPage.nextFrame()

    const aboveThresholdState = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        lowQuality: canvas.low_quality,
        scale: canvas.ds.scale
      }
    })

    // If still above threshold, should be high quality
    if (aboveThresholdState.scale > expectedThreshold) {
      expect(aboveThresholdState.lowQuality).toBe(false)
    }

    // Zoom out more to trigger LOD (below threshold)
    await comfyPage.canvasOps.zoom(120, 5) // Zoom out 5 more steps
    await comfyPage.nextFrame()

    // Check that LOD is now active
    const zoomedOutState = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        lowQuality: canvas.low_quality,
        scale: canvas.ds.scale
      }
    })

    expect(zoomedOutState.scale).toBeLessThan(expectedThreshold)
    expect(zoomedOutState.lowQuality).toBe(true)

    // Zoom back in to disable LOD (above threshold)
    await comfyPage.canvasOps.zoom(-120, 15) // Zoom in 15 steps
    await comfyPage.nextFrame()

    // Check that LOD is now inactive
    const zoomedInState = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        lowQuality: canvas.low_quality,
        scale: canvas.ds.scale
      }
    })

    expect(zoomedInState.scale).toBeGreaterThan(expectedThreshold)
    expect(zoomedInState.lowQuality).toBe(false)
  })

  test('Should update threshold when font size setting changes', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('default')

    // Change the font size setting to 14px (more aggressive LOD)
    await comfyPage.settings.setSetting(
      'LiteGraph.Canvas.MinFontSizeForLOD',
      14
    )

    // Check that font size updated
    const newState = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        minFontSize: canvas.min_font_size_for_lod
      }
    })

    expect(newState.minFontSize).toBe(14)
    // Expected threshold would be 14px / 14px = 1.0

    // At default zoom, LOD should still be inactive (scale is exactly 1.0, not less than)
    const lodState = await comfyPage.page.evaluate(() => {
      return window.app!.canvas.low_quality
    })
    expect(lodState).toBe(false)

    // Zoom out slightly to trigger LOD
    await comfyPage.canvasOps.zoom(120, 1) // Zoom out 1 step
    await comfyPage.nextFrame()

    const afterZoom = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        lowQuality: canvas.low_quality,
        scale: canvas.ds.scale
      }
    })

    expect(afterZoom.scale).toBeLessThan(1.0)
    expect(afterZoom.lowQuality).toBe(true)
  })

  test('Should disable LOD when font size is set to 0', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('default')

    // Disable LOD by setting font size to 0
    await comfyPage.settings.setSetting('LiteGraph.Canvas.MinFontSizeForLOD', 0)

    // Zoom out significantly
    await comfyPage.canvasOps.zoom(120, 20) // Zoom out 20 steps
    await comfyPage.nextFrame()

    // LOD should remain disabled even at very low zoom
    const state = await comfyPage.page.evaluate(() => {
      const canvas = window.app!.canvas
      return {
        lowQuality: canvas.low_quality,
        scale: canvas.ds.scale,
        minFontSize: canvas.min_font_size_for_lod
      }
    })

    expect(state.minFontSize).toBe(0) // LOD disabled
    expect(state.lowQuality).toBe(false)
    expect(state.scale).toBeLessThan(0.2) // Very zoomed out
  })

  test(
    'Should show visual difference between LOD on and off',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      // Load a workflow with text-heavy nodes for clear visual difference
      await comfyPage.workflow.loadWorkflow('default')

      // Set zoom level clearly below the threshold to ensure LOD activates
      const targetZoom = 0.4 // Well below default threshold of ~0.571

      // Zoom to target level
      await comfyPage.page.evaluate((zoom) => {
        window.app!.canvas.ds.scale = zoom
        window.app!.canvas.setDirty(true, true)
      }, targetZoom)
      await comfyPage.nextFrame()

      // Take snapshot with LOD active (default 8px setting)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'lod-comparison-low-quality.png'
      )

      const lowQualityState = await comfyPage.page.evaluate(() => {
        const canvas = window.app!.canvas
        return {
          lowQuality: canvas.low_quality,
          scale: canvas.ds.scale
        }
      })
      expect(lowQualityState.lowQuality).toBe(true)

      // Disable LOD to see high quality at same zoom
      await comfyPage.settings.setSetting(
        'LiteGraph.Canvas.MinFontSizeForLOD',
        0
      )
      await comfyPage.nextFrame()

      // Take snapshot with LOD disabled (full quality at same zoom)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'lod-comparison-high-quality.png'
      )

      const highQualityState = await comfyPage.page.evaluate(() => {
        const canvas = window.app!.canvas
        return {
          lowQuality: canvas.low_quality,
          scale: canvas.ds.scale
        }
      })
      expect(highQualityState.lowQuality).toBe(false)
      expect(highQualityState.scale).toBeCloseTo(targetZoom, 2)
    }
  )
})
