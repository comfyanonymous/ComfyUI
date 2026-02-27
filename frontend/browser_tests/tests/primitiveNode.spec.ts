import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import type { NodeReference } from '../fixtures/utils/litegraphUtils'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Primitive Node', { tag: ['@screenshot', '@node'] }, () => {
  test('Can load with correct size', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('primitive/primitive_node')
    await expect(comfyPage.canvas).toHaveScreenshot('primitive_node.png')
  })

  // When link is dropped on widget, it should automatically convert the widget
  // to input.
  test('Can connect to widget', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow(
      'primitive/primitive_node_unconnected'
    )
    const primitiveNode: NodeReference =
      await comfyPage.nodeOps.getNodeRefById(1)
    const ksamplerNode: NodeReference =
      await comfyPage.nodeOps.getNodeRefById(2)
    // Connect the output of the primitive node to the input of first widget of the ksampler node
    await primitiveNode.connectWidget(0, ksamplerNode, 0)
    await expect(comfyPage.canvas).toHaveScreenshot(
      'primitive_node_connected.png'
    )
  })

  test('Can connect to dom widget', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow(
      'primitive/primitive_node_unconnected_dom_widget'
    )
    const primitiveNode: NodeReference =
      await comfyPage.nodeOps.getNodeRefById(1)
    const clipEncoderNode: NodeReference =
      await comfyPage.nodeOps.getNodeRefById(2)
    await primitiveNode.connectWidget(0, clipEncoderNode, 0)
    await expect(comfyPage.canvas).toHaveScreenshot(
      'primitive_node_connected_dom_widget.png'
    )
  })

  test('Can connect to static primitive node', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow(
      'primitive/static_primitive_unconnected'
    )
    const primitiveNode: NodeReference =
      await comfyPage.nodeOps.getNodeRefById(1)
    const ksamplerNode: NodeReference =
      await comfyPage.nodeOps.getNodeRefById(2)
    await primitiveNode.connectWidget(0, ksamplerNode, 0)
    await expect(comfyPage.canvas).toHaveScreenshot(
      'static_primitive_connected.png'
    )
  })

  test('Report missing nodes when connect to missing node', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(
      'primitive/primitive_node_connect_missing_node'
    )
    // Wait for the element with the .comfy-missing-nodes selector to be visible
    const missingNodesWarning = comfyPage.page.locator('.comfy-missing-nodes')
    await expect(missingNodesWarning).toBeVisible()
  })
})
