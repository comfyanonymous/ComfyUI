import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import { recordMeasurement } from '../helpers/perfReporter'

test.describe('Performance', { tag: ['@perf'] }, () => {
  test('canvas idle style recalculations', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('default')
    await comfyPage.perf.startMeasuring()

    // Let the canvas idle for 2 seconds — no user interaction.
    // Measures baseline style recalcs from reactive state + render loop.
    for (let i = 0; i < 120; i++) {
      await comfyPage.nextFrame()
    }

    const m = await comfyPage.perf.stopMeasuring('canvas-idle')
    recordMeasurement(m)
    console.log(
      `Canvas idle: ${m.styleRecalcs} style recalcs, ${m.layouts} layouts`
    )
  })

  test('canvas mouse interaction style recalculations', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('default')
    await comfyPage.perf.startMeasuring()

    const canvas = comfyPage.canvas
    const box = await canvas.boundingBox()
    if (!box) throw new Error('Canvas bounding box not available')

    // Sweep mouse across the canvas — crosses nodes, empty space, slots
    for (let i = 0; i < 100; i++) {
      await comfyPage.page.mouse.move(
        box.x + (box.width * i) / 100,
        box.y + (box.height * (i % 3)) / 3
      )
    }

    const m = await comfyPage.perf.stopMeasuring('canvas-mouse-sweep')
    recordMeasurement(m)
    console.log(
      `Mouse sweep: ${m.styleRecalcs} style recalcs, ${m.layouts} layouts`
    )
  })

  test('DOM widget clipping during node selection', async ({ comfyPage }) => {
    // Load default workflow which has DOM widgets (text inputs, combos)
    await comfyPage.workflow.loadWorkflow('default')
    await comfyPage.perf.startMeasuring()

    // Select and deselect nodes rapidly to trigger clipping recalculation
    const canvas = comfyPage.canvas
    const box = await canvas.boundingBox()
    if (!box) throw new Error('Canvas bounding box not available')

    for (let i = 0; i < 20; i++) {
      // Click on canvas area (nodes occupy various positions)
      await comfyPage.page.mouse.click(
        box.x + box.width / 3 + (i % 5) * 30,
        box.y + box.height / 3 + (i % 4) * 30
      )
      await comfyPage.nextFrame()
    }

    const m = await comfyPage.perf.stopMeasuring('dom-widget-clipping')
    recordMeasurement(m)
    console.log(`Clipping: ${m.layouts} forced layouts`)
  })
})
