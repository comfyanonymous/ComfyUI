import { expect } from '@playwright/test'

import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'

import type { ComfyPage } from '../fixtures/ComfyPage'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import { TestIds } from '../fixtures/selectors'
import { fitToViewInstant } from '../helpers/fitToView'
import {
  getPromotedWidgetNames,
  getPromotedWidgetCount,
  getPromotedWidgets
} from '../helpers/promotedWidgets'

/**
 * Check whether we're currently in a subgraph.
 */
async function isInSubgraph(comfyPage: ComfyPage): Promise<boolean> {
  return comfyPage.page.evaluate(() => {
    const graph = window.app!.canvas.graph
    return !!graph && 'inputNode' in graph
  })
}

async function exitSubgraphViaBreadcrumb(comfyPage: ComfyPage): Promise<void> {
  const breadcrumb = comfyPage.page.getByTestId(TestIds.breadcrumb.subgraph)
  await breadcrumb.waitFor({ state: 'visible', timeout: 5000 })

  const parentLink = breadcrumb.getByRole('link').first()
  await expect(parentLink).toBeVisible()
  await parentLink.click()
  await comfyPage.nextFrame()
}

test.describe(
  'Subgraph Widget Promotion',
  { tag: ['@subgraph', '@widget'] },
  () => {
    test.describe('Auto-promotion on Convert to Subgraph', () => {
      test('Recommended widgets are auto-promoted when creating a subgraph', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow('default')

        // Select just the KSampler node (id 3) which has a "seed" widget
        const ksampler = await comfyPage.nodeOps.getNodeRefById('3')
        await ksampler.click('title')
        const subgraphNode = await ksampler.convertToSubgraph()
        await comfyPage.nextFrame()

        // SubgraphNode should exist
        expect(await subgraphNode.exists()).toBe(true)

        // The KSampler has a "seed" widget which is in the recommended list.
        // The promotion store should have at least the seed widget promoted.
        const nodeId = String(subgraphNode.id)
        const promotedNames = await getPromotedWidgetNames(comfyPage, nodeId)
        expect(promotedNames).toContain('seed')

        // SubgraphNode should have widgets (promoted views)
        const widgetCount = await getPromotedWidgetCount(comfyPage, nodeId)
        expect(widgetCount).toBeGreaterThan(0)
      })

      test('CLIPTextEncode text widget is auto-promoted', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow('default')

        // Select the positive CLIPTextEncode node (id 6)
        const clipNode = await comfyPage.nodeOps.getNodeRefById('6')
        await clipNode.click('title')
        const subgraphNode = await clipNode.convertToSubgraph()
        await comfyPage.nextFrame()

        const nodeId = String(subgraphNode.id)
        const promotedNames = await getPromotedWidgetNames(comfyPage, nodeId)
        expect(promotedNames.length).toBeGreaterThan(0)

        // CLIPTextEncode is in the recommendedNodes list, so its text widget
        // should be promoted
        expect(promotedNames).toContain('text')
      })

      test('SaveImage/PreviewImage nodes get pseudo-widget promoted', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow('default')
        await fitToViewInstant(comfyPage)

        // Select the SaveImage node (id 9 in default workflow)
        const saveNode = await comfyPage.nodeOps.getNodeRefById('9')
        await saveNode.click('title')
        const subgraphNode = await saveNode.convertToSubgraph()
        await comfyPage.nextFrame()

        const promotedNames = await getPromotedWidgetNames(
          comfyPage,
          String(subgraphNode.id)
        )

        // SaveImage is in the recommendedNodes list, so filename_prefix is promoted
        expect(promotedNames).toContain('filename_prefix')
      })
    })

    test.describe('Promoted Widget Visibility in LiteGraph Mode', () => {
      test('Promoted text widget is visible on SubgraphNode', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        // The subgraph node (id 11) should have a text widget promoted
        const textarea = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await expect(textarea).toBeVisible()
        await expect(textarea).toHaveCount(1)
      })

      test('Multiple promoted widgets all render on SubgraphNode', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-multiple-promoted-widgets'
        )
        await comfyPage.nextFrame()

        const textareas = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await expect(textareas.first()).toBeVisible()
        const count = await textareas.count()
        expect(count).toBeGreaterThan(1)
      })
    })

    test.describe('Promoted Widget Visibility in Vue Mode', () => {
      test.beforeEach(async ({ comfyPage }) => {
        await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
      })

      test('Promoted text widget renders on SubgraphNode in Vue mode', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.vueNodes.waitForNodes()

        // SubgraphNode (id 11) should render with its body
        const subgraphVueNode = comfyPage.vueNodes.getNodeLocator('11')
        await expect(subgraphVueNode).toBeVisible()

        // It should have the Enter Subgraph button
        const enterButton = subgraphVueNode.getByTestId('subgraph-enter-button')
        await expect(enterButton).toBeVisible()

        // The promoted text widget should render inside the node
        const nodeBody = subgraphVueNode.locator('[data-testid="node-body-11"]')
        await expect(nodeBody).toBeVisible()

        // Widgets section should exist and have at least one widget
        const widgets = nodeBody.locator('.lg-node-widgets > div')
        await expect(widgets.first()).toBeVisible()
      })

      test('Enter Subgraph button navigates into subgraph in Vue mode', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.vueNodes.waitForNodes()

        await comfyPage.vueNodes.enterSubgraph('11')
        await comfyPage.nextFrame()

        expect(await isInSubgraph(comfyPage)).toBe(true)
      })

      test('Multiple promoted widgets render on SubgraphNode in Vue mode', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-multiple-promoted-widgets'
        )
        await comfyPage.vueNodes.waitForNodes()

        const subgraphVueNode = comfyPage.vueNodes.getNodeLocator('11')
        await expect(subgraphVueNode).toBeVisible()

        const nodeBody = subgraphVueNode.locator('[data-testid="node-body-11"]')
        const widgets = nodeBody.locator('.lg-node-widgets > div')
        const count = await widgets.count()
        expect(count).toBeGreaterThan(1)
      })
    })

    test.describe('Promoted Widget Reactivity', () => {
      test('Value changes on promoted widget sync to interior widget', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        const testContent = 'promoted-value-sync-test'

        // Type into the promoted textarea on the SubgraphNode
        const textarea = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await textarea.fill(testContent)
        await comfyPage.nextFrame()

        // Navigate into subgraph
        const subgraphNode = await comfyPage.nodeOps.getNodeRefById('11')
        await subgraphNode.navigateIntoSubgraph()

        // Interior CLIPTextEncode textarea should have the same value
        const interiorTextarea = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await expect(interiorTextarea).toHaveValue(testContent)
      })

      test('Value changes on interior widget sync to promoted widget', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        const testContent = 'interior-value-sync-test'

        // Navigate into subgraph
        const subgraphNode = await comfyPage.nodeOps.getNodeRefById('11')
        await subgraphNode.navigateIntoSubgraph()

        // Type into the interior CLIPTextEncode textarea
        const interiorTextarea = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await interiorTextarea.fill(testContent)
        await comfyPage.nextFrame()

        // Navigate back to parent graph
        await exitSubgraphViaBreadcrumb(comfyPage)

        // Promoted textarea on SubgraphNode should have the same value
        const promotedTextarea = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await expect(promotedTextarea).toHaveValue(testContent)
      })

      test('Value persists through repeated navigation', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        const testContent = 'persistence-through-navigation'

        // Set value on promoted widget
        const textarea = comfyPage.page.getByTestId(
          TestIds.widgets.domWidgetTextarea
        )
        await textarea.fill(testContent)

        // Navigate in and out multiple times
        for (let i = 0; i < 3; i++) {
          const subgraphNode = await comfyPage.nodeOps.getNodeRefById('11')
          await subgraphNode.navigateIntoSubgraph()
          const interiorTextarea = comfyPage.page.getByTestId(
            TestIds.widgets.domWidgetTextarea
          )
          await expect(interiorTextarea).toHaveValue(testContent)

          await exitSubgraphViaBreadcrumb(comfyPage)

          const promotedTextarea = comfyPage.page.getByTestId(
            TestIds.widgets.domWidgetTextarea
          )
          await expect(promotedTextarea).toHaveValue(testContent)
        }
      })
    })

    test.describe('Manual Promote/Demote via Context Menu', () => {
      test.beforeEach(async ({ comfyPage }) => {
        await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
      })

      test('Can promote a widget from inside a subgraph', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow('subgraphs/basic-subgraph')

        const subgraphNode = await comfyPage.nodeOps.getNodeRefById('2')
        await subgraphNode.navigateIntoSubgraph()

        // Get the KSampler node (id 1) inside the subgraph
        const ksampler = await comfyPage.nodeOps.getNodeRefById('1')

        // Right-click on the KSampler's "steps" widget (index 2) to promote it
        const stepsWidget = await ksampler.getWidget(2)
        const widgetPos = await stepsWidget.getPosition()
        await comfyPage.canvas.click({
          position: widgetPos,
          button: 'right',
          force: true
        })
        await comfyPage.nextFrame()

        // Look for the Promote Widget menu entry
        const promoteEntry = comfyPage.page
          .locator('.litemenu-entry')
          .filter({ hasText: /Promote Widget/ })

        await expect(promoteEntry).toBeVisible()
        await promoteEntry.click()
        await comfyPage.nextFrame()

        // Navigate back to parent
        await exitSubgraphViaBreadcrumb(comfyPage)

        // SubgraphNode should now have the promoted widget
        const widgetCount = await getPromotedWidgetCount(comfyPage, '2')
        expect(widgetCount).toBeGreaterThan(0)
      })

      test('Can un-promote a widget from inside a subgraph', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow('subgraphs/basic-subgraph')

        // First promote a canvas-rendered widget (KSampler "steps")
        const subgraphNode = await comfyPage.nodeOps.getNodeRefById('2')
        await subgraphNode.navigateIntoSubgraph()

        const ksampler = await comfyPage.nodeOps.getNodeRefById('1')
        const stepsWidget = await ksampler.getWidget(2)
        const widgetPos = await stepsWidget.getPosition()

        await comfyPage.canvas.click({
          position: widgetPos,
          button: 'right',
          force: true
        })
        await comfyPage.nextFrame()

        const promoteEntry = comfyPage.page
          .locator('.litemenu-entry')
          .filter({ hasText: /Promote Widget/ })
        await expect(promoteEntry).toBeVisible()
        await promoteEntry.click()
        await comfyPage.nextFrame()

        // Navigate back and verify promotion took effect
        await exitSubgraphViaBreadcrumb(comfyPage)
        await fitToViewInstant(comfyPage)
        await comfyPage.nextFrame()

        const initialWidgetCount = await getPromotedWidgetCount(comfyPage, '2')
        expect(initialWidgetCount).toBeGreaterThan(0)

        // Navigate back in and un-promote
        const subgraphNode2 = await comfyPage.nodeOps.getNodeRefById('2')
        await subgraphNode2.navigateIntoSubgraph()
        const stepsWidget2 = await (
          await comfyPage.nodeOps.getNodeRefById('1')
        ).getWidget(2)
        const widgetPos2 = await stepsWidget2.getPosition()

        await comfyPage.canvas.click({
          position: widgetPos2,
          button: 'right',
          force: true
        })
        await comfyPage.nextFrame()

        const unpromoteEntry = comfyPage.page
          .locator('.litemenu-entry')
          .filter({ hasText: /Un-Promote Widget/ })

        await expect(unpromoteEntry).toBeVisible()
        await unpromoteEntry.click()
        await comfyPage.nextFrame()

        // Navigate back to parent
        await exitSubgraphViaBreadcrumb(comfyPage)

        // SubgraphNode should have fewer widgets
        const finalWidgetCount = await getPromotedWidgetCount(comfyPage, '2')
        expect(finalWidgetCount).toBeLessThan(initialWidgetCount)
      })
    })

    test.describe('Pseudo-Widget Promotion', () => {
      test('Promotion store tracks pseudo-widget entries for subgraph with preview node', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-preview-node'
        )
        await comfyPage.nextFrame()

        // The SaveImage node is in the recommendedNodes list, so its
        // filename_prefix widget should be auto-promoted
        const promotedNames = await getPromotedWidgetNames(comfyPage, '5')
        expect(promotedNames.length).toBeGreaterThan(0)
        expect(promotedNames).toContain('filename_prefix')
      })

      test('Converting SaveImage to subgraph promotes its widgets', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow('default')
        await fitToViewInstant(comfyPage)

        // Select SaveImage (id 9)
        const saveNode = await comfyPage.nodeOps.getNodeRefById('9')
        await saveNode.click('title')
        const subgraphNode = await saveNode.convertToSubgraph()
        await comfyPage.nextFrame()

        // SaveImage is a recommended node, so filename_prefix should be promoted
        const nodeId = String(subgraphNode.id)
        const promotedNames = await getPromotedWidgetNames(comfyPage, nodeId)
        expect(promotedNames.length).toBeGreaterThan(0)

        const widgetCount = await getPromotedWidgetCount(comfyPage, nodeId)
        expect(widgetCount).toBeGreaterThan(0)
      })
    })

    test.describe('Legacy And Round-Trip Coverage', () => {
      test('Legacy -1 proxyWidgets entries are hydrated to concrete interior node IDs', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-compressed-target-slot'
        )
        await comfyPage.nextFrame()

        const promotedWidgets = await getPromotedWidgets(comfyPage, '2')
        expect(promotedWidgets.length).toBeGreaterThan(0)
        expect(
          promotedWidgets.some(([interiorNodeId]) => interiorNodeId === '-1')
        ).toBe(false)
        expect(
          promotedWidgets.some(
            ([interiorNodeId, widgetName]) =>
              interiorNodeId !== '-1' && widgetName === 'batch_size'
          )
        ).toBe(true)
      })

      test('Promoted widgets survive serialize -> loadGraphData round-trip', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        const beforePromoted = await getPromotedWidgetNames(comfyPage, '11')
        expect(beforePromoted).toContain('text')

        const serialized = await comfyPage.page.evaluate(() => {
          return window.app!.graph!.serialize()
        })

        await comfyPage.page.evaluate((workflow: ComfyWorkflowJSON) => {
          return window.app!.loadGraphData(workflow)
        }, serialized as ComfyWorkflowJSON)
        await comfyPage.nextFrame()

        const afterPromoted = await getPromotedWidgetNames(comfyPage, '11')
        expect(afterPromoted).toContain('text')

        const widgetCount = await getPromotedWidgetCount(comfyPage, '11')
        expect(widgetCount).toBeGreaterThan(0)
      })

      test('Cloning a subgraph node keeps promoted widget entries on original and clone', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        const originalNode = await comfyPage.nodeOps.getNodeRefById('11')
        const originalPos = await originalNode.getPosition()

        await comfyPage.page.mouse.move(originalPos.x + 16, originalPos.y + 16)
        await comfyPage.page.keyboard.down('Alt')
        await comfyPage.page.mouse.down()
        await comfyPage.nextFrame()
        await comfyPage.page.mouse.move(originalPos.x + 72, originalPos.y + 72)
        await comfyPage.page.mouse.up()
        await comfyPage.page.keyboard.up('Alt')
        await comfyPage.nextFrame()

        const subgraphNodeIds = await comfyPage.page.evaluate(() => {
          const graph = window.app!.canvas.graph!
          return graph.nodes
            .filter(
              (n) =>
                typeof n.isSubgraphNode === 'function' && n.isSubgraphNode()
            )
            .map((n) => String(n.id))
        })

        expect(subgraphNodeIds.length).toBeGreaterThan(1)
        for (const nodeId of subgraphNodeIds) {
          const promotedWidgets = await getPromotedWidgets(comfyPage, nodeId)
          expect(promotedWidgets.length).toBeGreaterThan(0)
          expect(
            promotedWidgets.some(([, widgetName]) => widgetName === 'text')
          ).toBe(true)
        }
      })
    })

    test.describe('Vue Mode - Promoted Preview Content', () => {
      test.beforeEach(async ({ comfyPage }) => {
        await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
      })

      test('SubgraphNode with preview node shows hasCustomContent area in Vue mode', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-preview-node'
        )
        await comfyPage.vueNodes.waitForNodes()

        const subgraphVueNode = comfyPage.vueNodes.getNodeLocator('5')
        await expect(subgraphVueNode).toBeVisible()

        // The node body should exist
        const nodeBody = subgraphVueNode.locator('[data-testid="node-body-5"]')
        await expect(nodeBody).toBeVisible()
      })
    })

    test.describe('Promotion Cleanup', () => {
      test('Removing subgraph node clears promotion store entries', async ({
        comfyPage
      }) => {
        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )
        await comfyPage.nextFrame()

        // Verify promotions exist
        const namesBefore = await getPromotedWidgetNames(comfyPage, '11')
        expect(namesBefore.length).toBeGreaterThan(0)

        // Delete the subgraph node
        const subgraphNode = await comfyPage.nodeOps.getNodeRefById('11')
        await subgraphNode.click('title')
        await comfyPage.page.keyboard.press('Delete')
        await comfyPage.nextFrame()

        // Node no longer exists, so promoted widgets should be gone
        const nodeExists = await comfyPage.page.evaluate(() => {
          return !!window.app!.canvas.graph!.getNodeById('11')
        })
        expect(nodeExists).toBe(false)
      })

      test('Removing I/O slot removes associated promoted widget', async ({
        comfyPage
      }) => {
        await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')

        await comfyPage.workflow.loadWorkflow(
          'subgraphs/subgraph-with-promoted-text-widget'
        )

        const initialWidgetCount = await getPromotedWidgetCount(comfyPage, '11')
        expect(initialWidgetCount).toBeGreaterThan(0)

        // Navigate into subgraph
        const subgraphNode = await comfyPage.nodeOps.getNodeRefById('11')
        await subgraphNode.navigateIntoSubgraph()

        // Remove the text input slot
        await comfyPage.subgraph.rightClickInputSlot('text')
        await comfyPage.contextMenu.clickLitegraphMenuItem('Remove Slot')
        await comfyPage.nextFrame()

        // Navigate back via breadcrumb
        await comfyPage.page
          .getByTestId(TestIds.breadcrumb.subgraph)
          .waitFor({ state: 'visible', timeout: 5000 })
        const homeBreadcrumb = comfyPage.page.getByRole('link', {
          name: 'subgraph-with-promoted-text-widget'
        })
        await homeBreadcrumb.waitFor({ state: 'visible' })
        await homeBreadcrumb.click()
        await comfyPage.nextFrame()

        // Widget count should be reduced
        const finalWidgetCount = await getPromotedWidgetCount(comfyPage, '11')
        expect(finalWidgetCount).toBeLessThan(initialWidgetCount)
      })
    })
  }
)
