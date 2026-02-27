import type { Locator, Page } from '@playwright/test'

import type { NodeId } from '../../../../../src/platform/workflow/validation/schemas/workflowSchema'
import { getSlotKey } from '../../../../../src/renderer/core/layout/slots/slotIdentifier'
import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'
import { getMiddlePoint } from '../../../../fixtures/utils/litegraphUtils'
import { fitToViewInstant } from '../../../../helpers/fitToView'

async function getCenter(locator: Locator): Promise<{ x: number; y: number }> {
  const box = await locator.boundingBox()
  if (!box) throw new Error('Slot bounding box not available')
  return {
    x: box.x + box.width / 2,
    y: box.y + box.height / 2
  }
}

async function getInputLinkDetails(
  page: Page,
  nodeId: NodeId,
  slotIndex: number
) {
  return await page.evaluate(
    ([targetNodeId, targetSlot]) => {
      const app = window.app
      const graph = app?.canvas?.graph ?? app?.graph
      if (!graph) return null

      const node = graph.getNodeById(targetNodeId)
      if (!node) return null

      const input = node.inputs?.[targetSlot]
      if (!input) return null

      const linkId = input.link
      if (linkId == null) return null

      const link = graph.getLink?.(linkId)
      if (!link) return null

      return {
        id: link.id,
        originId: link.origin_id,
        originSlot:
          typeof link.origin_slot === 'string'
            ? Number.parseInt(link.origin_slot, 10)
            : link.origin_slot,
        targetId: link.target_id,
        targetSlot:
          typeof link.target_slot === 'string'
            ? Number.parseInt(link.target_slot, 10)
            : link.target_slot,
        parentId: link.parentId ?? null
      }
    },
    [nodeId, slotIndex] as const
  )
}

function slotLocator(
  page: Page,
  nodeId: NodeId,
  slotIndex: number,
  isInput: boolean
) {
  const key = getSlotKey(String(nodeId), slotIndex, isInput)
  return page.locator(`[data-slot-key="${key}"]`)
}

async function expectVisibleAll(...locators: Locator[]) {
  await Promise.all(locators.map((l) => expect(l).toBeVisible()))
}

async function getSlotCenter(
  page: Page,
  nodeId: NodeId,
  slotIndex: number,
  isInput: boolean
) {
  const locator = slotLocator(page, nodeId, slotIndex, isInput)
  await expect(locator).toBeVisible()
  return await getCenter(locator)
}

async function connectSlots(
  page: Page,
  from: { nodeId: NodeId; index: number },
  to: { nodeId: NodeId; index: number },
  nextFrame: () => Promise<void>
) {
  const fromLoc = slotLocator(page, from.nodeId, from.index, false)
  const toLoc = slotLocator(page, to.nodeId, to.index, true)
  await expectVisibleAll(fromLoc, toLoc)
  await fromLoc.dragTo(toLoc, { force: true })
  await nextFrame()
}

test.describe('Vue Node Link Interaction', { tag: '@screenshot' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.settings.setSetting('Comfy.NodeSearchBoxImpl', 'default')
    // await comfyPage.setup()
    await comfyPage.workflow.loadWorkflow('vueNodes/simple-triple')
    await comfyPage.vueNodes.waitForNodes()
    await fitToViewInstant(comfyPage)
  })

  test('should show a link dragging out from a slot when dragging on a slot', async ({
    comfyPage,
    comfyMouse
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(samplerNode).toBeTruthy()

    const slot = slotLocator(comfyPage.page, samplerNode.id, 0, false)
    await expect(slot).toBeVisible()

    const start = await getCenter(slot)

    // Arbitrary value
    const dragTarget = {
      x: start.x + 180,
      y: start.y - 140
    }

    await comfyMouse.move(start)
    await comfyMouse.drag(dragTarget)
    await comfyPage.nextFrame()

    try {
      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-dragging-link.png'
      )
    } finally {
      await comfyMouse.drop()
    }
  })

  test('should create a link when dropping on a compatible slot', async ({
    comfyPage
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const vaeNode = (await comfyPage.nodeOps.getNodeRefsByType('VAEDecode'))[0]
    expect(samplerNode && vaeNode).toBeTruthy()

    const samplerOutput = await samplerNode.getOutput(0)
    const vaeInput = await vaeNode.getInput(0)

    await connectSlots(
      comfyPage.page,
      { nodeId: samplerNode.id, index: 0 },
      { nodeId: vaeNode.id, index: 0 },
      () => comfyPage.nextFrame()
    )

    expect(await samplerOutput.getLinkCount()).toBe(1)
    expect(await vaeInput.getLinkCount()).toBe(1)

    const linkDetails = await getInputLinkDetails(comfyPage.page, vaeNode.id, 0)
    expect(linkDetails).not.toBeNull()
    expect(linkDetails).toMatchObject({
      originId: samplerNode.id,
      originSlot: 0,
      targetId: vaeNode.id,
      targetSlot: 0
    })
  })

  test('should not create a link when slot types are incompatible', async ({
    comfyPage
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const clipNode = (
      await comfyPage.nodeOps.getNodeRefsByType('CLIPTextEncode')
    )[0]
    expect(samplerNode && clipNode).toBeTruthy()

    const samplerOutput = await samplerNode.getOutput(0)
    const clipInput = await clipNode.getInput(0)

    const outputSlot = slotLocator(comfyPage.page, samplerNode.id, 0, false)
    const inputSlot = slotLocator(comfyPage.page, clipNode.id, 0, true)
    await expectVisibleAll(outputSlot, inputSlot)

    await outputSlot.dragTo(inputSlot, { force: true })
    await comfyPage.nextFrame()

    expect(await samplerOutput.getLinkCount()).toBe(0)
    expect(await clipInput.getLinkCount()).toBe(0)

    const graphLinkDetails = await getInputLinkDetails(
      comfyPage.page,
      clipNode.id,
      0
    )
    expect(graphLinkDetails).toBeNull()
  })

  test('should not create a link when dropping onto a slot on the same node', async ({
    comfyPage
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(samplerNode).toBeTruthy()

    const samplerOutput = await samplerNode.getOutput(0)
    const samplerInput = await samplerNode.getInput(3)

    const outputSlot = slotLocator(comfyPage.page, samplerNode.id, 0, false)
    const inputSlot = slotLocator(comfyPage.page, samplerNode.id, 3, true)
    await expectVisibleAll(outputSlot, inputSlot)

    await outputSlot.dragTo(inputSlot, { force: true })
    await comfyPage.nextFrame()

    expect(await samplerOutput.getLinkCount()).toBe(0)
    expect(await samplerInput.getLinkCount()).toBe(0)
  })

  test('should reuse the existing origin when dragging an input link', async ({
    comfyPage,
    comfyMouse
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const vaeNode = (await comfyPage.nodeOps.getNodeRefsByType('VAEDecode'))[0]
    expect(samplerNode && vaeNode).toBeTruthy()
    const samplerOutputCenter = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      0,
      false
    )
    const vaeInputCenter = await getSlotCenter(
      comfyPage.page,
      vaeNode.id,
      0,
      true
    )

    await comfyMouse.move(samplerOutputCenter)
    await comfyMouse.drag(vaeInputCenter)
    await comfyMouse.drop()

    const dragTarget = {
      x: vaeInputCenter.x + 160,
      y: vaeInputCenter.y - 100
    }

    await comfyMouse.move(vaeInputCenter)
    await comfyMouse.drag(dragTarget)
    await expect(comfyPage.canvas).toHaveScreenshot(
      'vue-node-input-drag-reuses-origin.png'
    )
    await comfyMouse.drop()
  })

  test('ctrl+alt drag from an input starts a fresh link', async ({
    comfyPage,
    comfyMouse
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const vaeNode = (await comfyPage.nodeOps.getNodeRefsByType('VAEDecode'))[0]
    expect(samplerNode && vaeNode).toBeTruthy()

    const samplerOutput = await samplerNode.getOutput(0)
    const vaeInput = await vaeNode.getInput(0)

    const samplerOutputCenter = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      0,
      false
    )
    const vaeInputCenter = await getSlotCenter(
      comfyPage.page,
      vaeNode.id,
      0,
      true
    )

    await comfyMouse.move(samplerOutputCenter)
    await comfyMouse.drag(vaeInputCenter)
    await comfyMouse.drop()

    await comfyPage.nextFrame()

    const dragTarget = {
      x: vaeInputCenter.x + 140,
      y: vaeInputCenter.y - 110
    }

    await comfyMouse.move(vaeInputCenter)
    await comfyPage.page.keyboard.down('Control')
    await comfyPage.page.keyboard.down('Alt')

    try {
      await comfyMouse.drag(dragTarget)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-input-drag-ctrl-alt.png'
      )
    } finally {
      await comfyMouse.drop().catch(() => {})
      await comfyPage.page.keyboard.up('Alt').catch(() => {})
      await comfyPage.page.keyboard.up('Control').catch(() => {})
    }

    await comfyPage.nextFrame()

    // Tcehnically intended to disconnect existing as well
    expect(await vaeInput.getLinkCount()).toBe(0)
    expect(await samplerOutput.getLinkCount()).toBe(0)
  })

  test('dropping an input link back on its slot restores the original connection', async ({
    comfyPage,
    comfyMouse
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const vaeNode = (await comfyPage.nodeOps.getNodeRefsByType('VAEDecode'))[0]
    expect(samplerNode && vaeNode).toBeTruthy()

    const samplerOutput = await samplerNode.getOutput(0)
    const vaeInput = await vaeNode.getInput(0)

    const samplerOutputCenter = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      0,
      false
    )
    const vaeInputCenter = await getSlotCenter(
      comfyPage.page,
      vaeNode.id,
      0,
      true
    )

    await comfyMouse.move(samplerOutputCenter)
    try {
      await comfyMouse.drag(vaeInputCenter)
    } finally {
      await comfyMouse.drop()
    }

    await comfyPage.nextFrame()

    const originalLink = await getInputLinkDetails(
      comfyPage.page,
      vaeNode.id,
      0
    )
    expect(originalLink).not.toBeNull()

    const dragTarget = {
      x: vaeInputCenter.x + 150,
      y: vaeInputCenter.y - 100
    }

    // To prevent needing a screenshot expectation for whether the link's off
    const vaeInputLocator = slotLocator(comfyPage.page, vaeNode.id, 0, true)
    const inputBox = await vaeInputLocator.boundingBox()
    if (!inputBox) throw new Error('Input slot bounding box not available')
    const isOutsideX =
      dragTarget.x < inputBox.x || dragTarget.x > inputBox.x + inputBox.width
    const isOutsideY =
      dragTarget.y < inputBox.y || dragTarget.y > inputBox.y + inputBox.height
    expect(isOutsideX || isOutsideY).toBe(true)

    await comfyMouse.move(vaeInputCenter)
    await comfyMouse.drag(dragTarget)
    await comfyMouse.move(vaeInputCenter)
    await comfyMouse.drop()

    await comfyPage.nextFrame()

    const restoredLink = await getInputLinkDetails(
      comfyPage.page,
      vaeNode.id,
      0
    )

    expect(restoredLink).not.toBeNull()
    if (!restoredLink || !originalLink) {
      throw new Error('Expected both original and restored links to exist')
    }
    expect(restoredLink).toMatchObject({
      originId: originalLink.originId,
      originSlot: originalLink.originSlot,
      targetId: originalLink.targetId,
      targetSlot: originalLink.targetSlot,
      parentId: originalLink.parentId
    })
    expect(await samplerOutput.getLinkCount()).toBe(1)
    expect(await vaeInput.getLinkCount()).toBe(1)
  })

  test('rerouted input drag preview remains anchored to reroute', async ({
    comfyPage,
    comfyMouse
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const vaeNode = (await comfyPage.nodeOps.getNodeRefsByType('VAEDecode'))[0]

    const samplerOutput = await samplerNode.getOutput(0)
    const vaeInput = await vaeNode.getInput(0)

    await connectSlots(
      comfyPage.page,
      { nodeId: samplerNode.id, index: 0 },
      { nodeId: vaeNode.id, index: 0 },
      () => comfyPage.nextFrame()
    )

    const outputPosition = await samplerOutput.getPosition()
    const inputPosition = await vaeInput.getPosition()
    const reroutePoint = getMiddlePoint(outputPosition, inputPosition)

    // Insert a reroute programmatically on the existing link between sampler output[0] and VAE input[0].
    // This avoids relying on an exact path hit-test position.
    await comfyPage.page.evaluate(
      ([targetNodeId, targetSlot, clientPoint]) => {
        const app = window.app
        const graph = app?.canvas?.graph ?? app?.graph
        if (!graph) throw new Error('Graph not available')
        const node = graph.getNodeById(targetNodeId)
        if (!node) throw new Error('Target node not found')
        const input = node.inputs?.[targetSlot]
        if (!input) throw new Error('Target input slot not found')

        const linkId = input.link
        if (linkId == null) throw new Error('Expected existing link on input')
        const link = graph.getLink(linkId)
        if (!link) throw new Error('Link not found')

        // Convert the client/canvas pixel coordinates to graph space
        const pos = app!.canvas.ds.convertCanvasToOffset([
          clientPoint.x,
          clientPoint.y
        ])
        graph.createReroute(pos, link)
      },
      [vaeNode.id, 0, reroutePoint] as const
    )

    await comfyPage.nextFrame()

    const vaeInputCenter = await getSlotCenter(
      comfyPage.page,
      vaeNode.id,
      0,
      true
    )
    const dragTarget = {
      x: vaeInputCenter.x + 160,
      y: vaeInputCenter.y - 120
    }

    let dropped = false
    try {
      await comfyMouse.move(vaeInputCenter)
      await comfyMouse.drag(dragTarget)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-reroute-input-drag.png'
      )
      await comfyMouse.move(vaeInputCenter)
      await comfyMouse.drop()
      dropped = true
    } finally {
      if (!dropped) {
        await comfyMouse.drop().catch(() => {})
      }
    }

    await comfyPage.nextFrame()

    const linkDetails = await getInputLinkDetails(comfyPage.page, vaeNode.id, 0)
    expect(linkDetails).not.toBeNull()
    expect(linkDetails?.originId).toBe(samplerNode.id)
    expect(linkDetails?.parentId).not.toBeNull()
  })

  test('rerouted output shift-drag preview remains anchored to reroute', async ({
    comfyPage,
    comfyMouse
  }) => {
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    const vaeNode = (await comfyPage.nodeOps.getNodeRefsByType('VAEDecode'))[0]
    expect(samplerNode && vaeNode).toBeTruthy()

    const samplerOutput = await samplerNode.getOutput(0)
    const vaeInput = await vaeNode.getInput(0)

    await connectSlots(
      comfyPage.page,
      { nodeId: samplerNode.id, index: 0 },
      { nodeId: vaeNode.id, index: 0 },
      () => comfyPage.nextFrame()
    )

    const outputPosition = await samplerOutput.getPosition()
    const inputPosition = await vaeInput.getPosition()
    const reroutePoint = getMiddlePoint(outputPosition, inputPosition)

    // Insert a reroute programmatically on the existing link between sampler output[0] and VAE input[0].
    // This avoids relying on an exact path hit-test position.
    await comfyPage.page.evaluate(
      ([targetNodeId, targetSlot, clientPoint]) => {
        const app = window.app
        const graph = app?.canvas?.graph ?? app?.graph
        if (!graph) throw new Error('Graph not available')
        const node = graph.getNodeById(targetNodeId)
        if (!node) throw new Error('Target node not found')
        const input = node.inputs?.[targetSlot]
        if (!input) throw new Error('Target input slot not found')

        const linkId = input.link
        if (linkId == null) throw new Error('Expected existing link on input')
        const link = graph.getLink(linkId)
        if (!link) throw new Error('Link not found')

        // Convert the client/canvas pixel coordinates to graph space
        const pos = app!.canvas.ds.convertCanvasToOffset([
          clientPoint.x,
          clientPoint.y
        ])
        graph.createReroute(pos, link)
      },
      [vaeNode.id, 0, reroutePoint] as const
    )

    await comfyPage.nextFrame()

    const outputCenter = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      0,
      false
    )
    const dragTarget = {
      x: outputCenter.x + 150,
      y: outputCenter.y - 140
    }

    let dropPending = false
    let shiftHeld = false
    try {
      await comfyMouse.move(outputCenter)
      await comfyPage.page.keyboard.down('Shift')
      shiftHeld = true
      dropPending = true
      await comfyMouse.drag(dragTarget)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-reroute-output-shift-drag.png'
      )
      await comfyMouse.move(outputCenter)
      await comfyMouse.drop()
      dropPending = false
    } finally {
      if (dropPending) await comfyMouse.drop().catch(() => {})
      if (shiftHeld) await comfyPage.page.keyboard.up('Shift').catch(() => {})
    }

    await comfyPage.nextFrame()

    const linkDetails = await getInputLinkDetails(comfyPage.page, vaeNode.id, 0)
    expect(linkDetails).not.toBeNull()
    expect(linkDetails?.originId).toBe(samplerNode.id)
    expect(linkDetails?.parentId).not.toBeNull()
  })

  test('dragging input to input drags existing link', async ({
    comfyPage,
    comfyMouse
  }) => {
    const clipNode = (
      await comfyPage.nodeOps.getNodeRefsByType('CLIPTextEncode')
    )[0]
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(clipNode && samplerNode).toBeTruthy()

    // Step 1: Connect CLIP's only output (index 0) to KSampler's second input (index 1)
    await connectSlots(
      comfyPage.page,
      { nodeId: clipNode.id, index: 0 },
      { nodeId: samplerNode.id, index: 1 },
      () => comfyPage.nextFrame()
    )

    // Verify initial link exists between CLIP -> KSampler input[1]
    const initialLink = await getInputLinkDetails(
      comfyPage.page,
      samplerNode.id,
      1
    )
    expect(initialLink).not.toBeNull()
    expect(initialLink).toMatchObject({
      originId: clipNode.id,
      targetId: samplerNode.id,
      targetSlot: 1
    })

    // Step 2: Drag from KSampler's second input to its third input (index 2)
    const input2Center = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      1,
      true
    )
    const input3Center = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      2,
      true
    )

    await comfyMouse.move(input2Center)
    await comfyMouse.drag(input3Center)
    await comfyMouse.drop()
    await comfyPage.nextFrame()

    // Expect old link removed from input[1]
    const afterSecondInput = await getInputLinkDetails(
      comfyPage.page,
      samplerNode.id,
      1
    )
    expect(afterSecondInput).toBeNull()

    // Expect new link exists at input[2] from CLIP
    const afterThirdInput = await getInputLinkDetails(
      comfyPage.page,
      samplerNode.id,
      2
    )
    expect(afterThirdInput).not.toBeNull()
    expect(afterThirdInput).toMatchObject({
      originId: clipNode.id,
      targetId: samplerNode.id,
      targetSlot: 2
    })
  })

  test('shift-dragging an output with multiple links should drag all links', async ({
    comfyPage,
    comfyMouse
  }) => {
    const clipNode = (
      await comfyPage.nodeOps.getNodeRefsByType('CLIPTextEncode')
    )[0]
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(clipNode && samplerNode).toBeTruthy()

    const clipOutput = await clipNode.getOutput(0)

    // Connect output[0] -> inputs[1] and [2]
    await connectSlots(
      comfyPage.page,
      { nodeId: clipNode.id, index: 0 },
      { nodeId: samplerNode.id, index: 1 },
      () => comfyPage.nextFrame()
    )
    await connectSlots(
      comfyPage.page,
      { nodeId: clipNode.id, index: 0 },
      { nodeId: samplerNode.id, index: 2 },
      () => comfyPage.nextFrame()
    )

    expect(await clipOutput.getLinkCount()).toBe(2)

    const outputCenter = await getSlotCenter(
      comfyPage.page,
      clipNode.id,
      0,
      false
    )
    const dragTarget = {
      x: outputCenter.x + 40,
      y: outputCenter.y - 140
    }

    let dropPending = false
    let shiftHeld = false
    try {
      await comfyMouse.move(outputCenter)
      await comfyPage.page.keyboard.down('Shift')
      shiftHeld = true
      await comfyMouse.drag(dragTarget)
      dropPending = true

      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-shift-output-multi-link.png'
      )
    } finally {
      if (dropPending) await comfyMouse.drop().catch(() => {})
      if (shiftHeld) await comfyPage.page.keyboard.up('Shift').catch(() => {})
    }
  })

  test('should snap to node center while dragging and link on drop', async ({
    comfyPage,
    comfyMouse
  }) => {
    const clipNode = (
      await comfyPage.nodeOps.getNodeRefsByType('CLIPTextEncode')
    )[0]
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(clipNode && samplerNode).toBeTruthy()

    // Start drag from CLIP output[0]
    const clipOutputCenter = await getSlotCenter(
      comfyPage.page,
      clipNode.id,
      0,
      false
    )

    // Drag to the visual center of the KSampler Vue node (not a slot)
    const samplerVue = comfyPage.vueNodes.getNodeLocator(String(samplerNode.id))
    await expect(samplerVue).toBeVisible()
    const samplerCenter = await getCenter(samplerVue)

    await comfyMouse.move(clipOutputCenter)
    await comfyMouse.drag(samplerCenter)

    // During drag, the preview should snap/highlight a compatible input on KSampler
    await expect(comfyPage.canvas).toHaveScreenshot('vue-node-snap-to-node.png')

    // Drop to create the link
    await comfyMouse.drop()
    await comfyPage.nextFrame()

    // Validate a link was created to one of KSampler's compatible inputs (1 or 2)
    const linkOnInput1 = await getInputLinkDetails(
      comfyPage.page,
      samplerNode.id,
      1
    )
    const linkOnInput2 = await getInputLinkDetails(
      comfyPage.page,
      samplerNode.id,
      2
    )

    const linked = linkOnInput1 ?? linkOnInput2
    expect(linked).not.toBeNull()
    expect(linked?.originId).toBe(clipNode.id)
    expect(linked?.targetId).toBe(samplerNode.id)
  })

  test('should snap to a specific compatible slot when targeting it', async ({
    comfyPage,
    comfyMouse
  }) => {
    const clipNode = (
      await comfyPage.nodeOps.getNodeRefsByType('CLIPTextEncode')
    )[0]
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(clipNode && samplerNode).toBeTruthy()

    // Drag from CLIP output[0] to KSampler input[2] (third slot) which is the
    // second compatible input for CLIP
    const clipOutputCenter = await getSlotCenter(
      comfyPage.page,
      clipNode.id,
      0,
      false
    )
    const samplerInput3Center = await getSlotCenter(
      comfyPage.page,
      samplerNode.id,
      2,
      true
    )

    await comfyMouse.move(clipOutputCenter)
    await comfyMouse.drag(samplerInput3Center)

    // Expect the preview to show snapping to the targeted slot
    await expect(comfyPage.canvas).toHaveScreenshot('vue-node-snap-to-slot.png')

    // Finish the connection
    await comfyMouse.drop()
    await comfyPage.nextFrame()

    const linkDetails = await getInputLinkDetails(
      comfyPage.page,
      samplerNode.id,
      2
    )
    expect(linkDetails).not.toBeNull()
    expect(linkDetails).toMatchObject({
      originId: clipNode.id,
      targetId: samplerNode.id,
      targetSlot: 2
    })
  })

  test('should batch disconnect all links with ctrl+alt+click on slot', async ({
    comfyPage
  }) => {
    const clipNode = (
      await comfyPage.nodeOps.getNodeRefsByType('CLIPTextEncode')
    )[0]
    const samplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )[0]
    expect(clipNode && samplerNode).toBeTruthy()

    await connectSlots(
      comfyPage.page,
      { nodeId: clipNode.id, index: 0 },
      { nodeId: samplerNode.id, index: 1 },
      () => comfyPage.nextFrame()
    )
    await connectSlots(
      comfyPage.page,
      { nodeId: clipNode.id, index: 0 },
      { nodeId: samplerNode.id, index: 2 },
      () => comfyPage.nextFrame()
    )

    const clipOutput = await clipNode.getOutput(0)
    expect(await clipOutput.getLinkCount()).toBe(2)

    const clipOutputSlot = slotLocator(comfyPage.page, clipNode.id, 0, false)

    await clipOutputSlot.dispatchEvent('pointerdown', {
      button: 0,
      buttons: 1,
      ctrlKey: true,
      altKey: true,
      shiftKey: false,
      bubbles: true,
      cancelable: true
    })
    await comfyPage.nextFrame()

    expect(await clipOutput.getLinkCount()).toBe(0)
  })

  test.describe('Release actions (Shift-drop)', () => {
    test('Context menu opens and endpoint is pinned on Shift-drop', async ({
      comfyPage,
      comfyMouse
    }) => {
      await comfyPage.settings.setSetting(
        'Comfy.LinkRelease.ActionShift',
        'context menu'
      )

      const samplerNode = (
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      )[0]
      expect(samplerNode).toBeTruthy()

      const outputCenter = await getSlotCenter(
        comfyPage.page,
        samplerNode.id,
        0,
        false
      )

      const dropPos = { x: outputCenter.x + 90, y: outputCenter.y - 70 }

      await comfyMouse.move(outputCenter)
      await comfyPage.page.keyboard.down('Shift')
      try {
        await comfyMouse.drag(dropPos)
        await comfyMouse.drop()
      } finally {
        await comfyPage.page.keyboard.up('Shift').catch(() => {})
      }

      // Context menu should be visible
      const contextMenu = comfyPage.page.locator('.litecontextmenu')
      await expect(contextMenu).toBeVisible()

      // Pinned endpoint should not change with mouse movement while menu is open
      const before = await comfyPage.page.evaluate(() => {
        const snap = window.app?.canvas?.linkConnector?.state?.snapLinksPos
        return Array.isArray(snap) ? [snap[0], snap[1]] : null
      })
      expect(before).not.toBeNull()

      // Move mouse elsewhere and verify snap position is unchanged
      await comfyMouse.move({ x: dropPos.x + 160, y: dropPos.y + 100 })
      const after = await comfyPage.page.evaluate(() => {
        const snap = window.app?.canvas?.linkConnector?.state?.snapLinksPos
        return Array.isArray(snap) ? [snap[0], snap[1]] : null
      })
      expect(after).toEqual(before)
    })

    test('Context menu -> Search pre-filters by link type and connects after selection', async ({
      comfyPage,
      comfyMouse
    }) => {
      await comfyPage.settings.setSetting(
        'Comfy.LinkRelease.ActionShift',
        'context menu'
      )
      await comfyPage.settings.setSetting(
        'Comfy.NodeSearchBoxImpl',
        'v1 (legacy)'
      )

      const samplerNode = (
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      )[0]
      expect(samplerNode).toBeTruthy()

      const outputCenter = await getSlotCenter(
        comfyPage.page,
        samplerNode.id,
        0,
        false
      )
      const dropPos = { x: outputCenter.x + 200, y: outputCenter.y - 100 }

      await comfyMouse.move(outputCenter)
      await comfyPage.page.keyboard.down('Shift')
      try {
        await comfyMouse.drag(dropPos)
        await comfyMouse.drop()
      } finally {
        await comfyPage.page.keyboard.up('Shift').catch(() => {})
      }

      // Open Search from the context menu
      await comfyPage.contextMenu.clickMenuItem('Search')
      await comfyPage.nextFrame()

      // Search box opens with prefilled type filter based on link type (LATENT)
      await expect(comfyPage.searchBox.input).toBeVisible()
      const chips = comfyPage.searchBox.filterChips
      // Ensure at least one filter chip exists and it matches the link type
      const chipCount = await chips.count()
      expect(chipCount).toBeGreaterThan(0)
      await expect(chips.first()).toContainText('LATENT')

      // Choose a compatible node and verify it auto-connects
      await comfyPage.searchBox.fillAndSelectFirstNode('VAEDecode')
      await comfyPage.nextFrame()

      // KSampler output should now have an outgoing link
      const samplerOutput = await samplerNode.getOutput(0)
      expect(await samplerOutput.getLinkCount()).toBe(1)

      // One of the VAEDecode nodes should have an incoming link on input[0]
      const vaeNodes = await comfyPage.nodeOps.getNodeRefsByType('VAEDecode')
      let linked = false
      for (const vae of vaeNodes) {
        const details = await getInputLinkDetails(comfyPage.page, vae.id, 0)
        if (details) {
          expect(details.originId).toBe(samplerNode.id)
          linked = true
          break
        }
      }
      expect(linked).toBe(true)
    })

    test('Search box opens on Shift-drop and connects after selection', async ({
      comfyPage,
      comfyMouse
    }) => {
      await comfyPage.settings.setSetting(
        'Comfy.LinkRelease.ActionShift',
        'search box'
      )
      await comfyPage.settings.setSetting(
        'Comfy.NodeSearchBoxImpl',
        'v1 (legacy)'
      )

      const samplerNode = (
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      )[0]
      expect(samplerNode).toBeTruthy()

      const outputCenter = await getSlotCenter(
        comfyPage.page,
        samplerNode.id,
        0,
        false
      )
      const dropPos = { x: outputCenter.x + 140, y: outputCenter.y - 100 }

      await comfyMouse.move(outputCenter)
      await comfyPage.page.keyboard.down('Shift')
      try {
        await comfyMouse.drag(dropPos)
        await comfyMouse.drop()
      } finally {
        await comfyPage.page.keyboard.up('Shift').catch(() => {})
      }

      // Search box should open directly
      await expect(comfyPage.searchBox.input).toBeVisible()
      await expect(comfyPage.searchBox.filterChips.first()).toContainText(
        'LATENT'
      )

      // Select a compatible node and verify connection
      await comfyPage.searchBox.fillAndSelectFirstNode('VAEDecode')
      await comfyPage.nextFrame()

      const samplerOutput = await samplerNode.getOutput(0)
      expect(await samplerOutput.getLinkCount()).toBe(1)

      const vaeNodes = await comfyPage.nodeOps.getNodeRefsByType('VAEDecode')
      let linked = false
      for (const vae of vaeNodes) {
        const details = await getInputLinkDetails(comfyPage.page, vae.id, 0)
        if (details) {
          expect(details.originId).toBe(samplerNode.id)
          linked = true
          break
        }
      }
      expect(linked).toBe(true)
    })
  })

  test('Dragging from subgraph input connects to correct slot', async ({
    comfyPage,
    comfyMouse
  }) => {
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )

    // Setup workflow with a KSampler node
    await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')
    await comfyPage.nodeOps.waitForGraphNodes(0)
    await comfyPage.command.executeCommand('Workspace.SearchBox.Toggle')
    await comfyPage.nextFrame()
    await comfyPage.searchBox.fillAndSelectFirstNode('KSampler')
    await comfyPage.nodeOps.waitForGraphNodes(1)

    // Convert the KSampler node to a subgraph
    let ksamplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler')
    )?.[0]
    await comfyPage.vueNodes.selectNode(String(ksamplerNode.id))
    await comfyPage.command.executeCommand('Comfy.Graph.ConvertToSubgraph')

    // Enter the subgraph
    await comfyPage.vueNodes.enterSubgraph()
    await fitToViewInstant(comfyPage)

    // Get the KSampler node inside the subgraph
    ksamplerNode = (
      await comfyPage.nodeOps.getNodeRefsByType('KSampler', true)
    )?.[0]
    const positiveInput = await ksamplerNode.getInput(1)
    const negativeInput = await ksamplerNode.getInput(2)

    const positiveInputPos = await getSlotCenter(
      comfyPage.page,
      ksamplerNode.id,
      1,
      true
    )

    const sourceSlot = await comfyPage.subgraph.getInputSlot()
    const calculatedSourcePos = await sourceSlot.getOpenSlotPosition()

    await comfyMouse.move(calculatedSourcePos)
    await comfyMouse.drag(positiveInputPos)
    await comfyMouse.drop()

    // Verify connection went to the correct slot
    const positiveLinks = await positiveInput.getLinkCount()
    const negativeLinks = await negativeInput.getLinkCount()
    expect(positiveLinks).toBe(1)
    expect(negativeLinks).toBe(0)
  })
})
