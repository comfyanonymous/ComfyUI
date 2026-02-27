import { expect } from '@playwright/test'

import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'

import type { ComfyPage } from '../fixtures/ComfyPage'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import type { NodeLibrarySidebarTab } from '../fixtures/components/SidebarTab'
import { DefaultGraphPositions } from '../fixtures/constants/defaultGraphPositions'
import type { NodeReference } from '../fixtures/utils/litegraphUtils'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
  await comfyPage.settings.setSetting('Comfy.NodeLibrary.NewDesign', false)
  await comfyPage.settings.setSetting('Comfy.NodeSearchBoxImpl', 'v1 (legacy)')
})

test.describe('Group Node', { tag: '@node' }, () => {
  test.describe('Node library sidebar', () => {
    const groupNodeName = 'DefautWorkflowGroupNode'
    const groupNodeCategory = 'group nodes>workflow'
    const groupNodeBookmarkName = `workflow>${groupNodeName}`
    let libraryTab: NodeLibrarySidebarTab

    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
      libraryTab = comfyPage.menu.nodeLibraryTab
      await comfyPage.nodeOps.convertAllNodesToGroupNode(groupNodeName)
      await libraryTab.open()
    })

    test('Is added to node library sidebar', async ({
      comfyPage: _comfyPage
    }) => {
      expect(await libraryTab.getFolder(groupNodeCategory).count()).toBe(1)
    })

    test('Can be added to canvas using node library sidebar', async ({
      comfyPage
    }) => {
      const initialNodeCount = await comfyPage.nodeOps.getGraphNodesCount()

      // Add group node from node library sidebar
      await libraryTab.getFolder(groupNodeCategory).click()
      await libraryTab.getNode(groupNodeName).click()

      // Verify the node is added to the canvas
      expect(await comfyPage.nodeOps.getGraphNodesCount()).toBe(
        initialNodeCount + 1
      )
    })

    test('Can be bookmarked and unbookmarked', async ({ comfyPage }) => {
      await libraryTab.getFolder(groupNodeCategory).click()
      await libraryTab
        .getNode(groupNodeName)
        .locator('.bookmark-button')
        .click()

      // Verify the node is added to the bookmarks tab
      expect(
        await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
      ).toEqual([groupNodeBookmarkName])
      // Verify the bookmark node with the same name is added to the tree
      expect(await libraryTab.getNode(groupNodeName).count()).not.toBe(0)

      // Unbookmark the node
      await libraryTab
        .getNode(groupNodeName)
        .locator('.bookmark-button')
        .first()
        .click()

      // Verify the node is removed from the bookmarks tab
      expect(
        await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
      ).toHaveLength(0)
    })

    test('Displays preview on bookmark hover', async ({ comfyPage }) => {
      await libraryTab.getFolder(groupNodeCategory).click()
      await libraryTab
        .getNode(groupNodeName)
        .locator('.bookmark-button')
        .click()
      await comfyPage.page.hover('.p-tree-node-label.tree-explorer-node-label')
      expect(await comfyPage.page.isVisible('.node-lib-node-preview')).toBe(
        true
      )
      await libraryTab
        .getNode(groupNodeName)
        .locator('.bookmark-button')
        .first()
        .click()
    })
  })
  // The 500ms fixed delay on the search results is causing flakiness
  // Potential solution: add a spinner state when the search is in progress,
  // and observe that state from the test. Blocker: the PrimeVue AutoComplete
  // does not have a v-model on the query, so we cannot observe the raw
  // query update, and thus cannot set the spinning state between the raw query
  // update and the debounced search update.
  test.skip(
    'Can be added to canvas using search',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      const groupNodeName = 'DefautWorkflowGroupNode'
      await comfyPage.nodeOps.convertAllNodesToGroupNode(groupNodeName)
      await comfyPage.canvasOps.doubleClick()
      await comfyPage.nextFrame()
      await comfyPage.searchBox.fillAndSelectFirstNode(groupNodeName)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'group-node-copy-added-from-search.png'
      )
    }
  )

  test('Displays tooltip on title hover', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.EnableTooltips', true)
    await comfyPage.nodeOps.convertAllNodesToGroupNode('Group Node')
    await comfyPage.page.mouse.move(47, 173)
    await expect(comfyPage.page.locator('.node-tooltip')).toBeVisible()
  })

  test('Manage group opens with the correct group selected', async ({
    comfyPage
  }) => {
    const makeGroup = async (name: string, type1: string, type2: string) => {
      const node1 = (await comfyPage.nodeOps.getNodeRefsByType(type1))[0]
      const node2 = (await comfyPage.nodeOps.getNodeRefsByType(type2))[0]
      await node1.click('title')
      await node2.click('title', {
        modifiers: ['Shift']
      })
      return await node2.convertToGroupNode(name)
    }

    const group1 = await makeGroup(
      'g1',
      'CLIPTextEncode',
      'CheckpointLoaderSimple'
    )
    const group2 = await makeGroup('g2', 'EmptyLatentImage', 'KSampler')

    const manage1 = await group1.manageGroupNode()
    await comfyPage.nextFrame()
    expect(await manage1.getSelectedNodeType()).toBe('g1')
    await manage1.close()
    await expect(manage1.root).not.toBeVisible()

    const manage2 = await group2.manageGroupNode()
    expect(await manage2.getSelectedNodeType()).toBe('g2')
  })

  test('Preserves hidden input configuration when containing duplicate node types', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(
      'groupnodes/group_node_identical_nodes_hidden_inputs'
    )
    await comfyPage.nextFrame()

    const groupNodeId = 19
    const groupNodeName = 'two_VAE_decode'

    const totalInputCount = await comfyPage.page.evaluate((nodeName) => {
      const {
        extra: { groupNodes }
      } = window.app!.graph!
      const { nodes } = groupNodes![nodeName]
      return nodes.reduce((acc, node) => acc + (node.inputs?.length ?? 0), 0)
    }, groupNodeName)

    const visibleInputCount = await comfyPage.page.evaluate((id) => {
      const node = window.app!.graph!.getNodeById(id)
      return node!.inputs.length
    }, groupNodeId)

    // Verify there are 4 total inputs (2 VAE decode nodes with 2 inputs each)
    expect(totalInputCount).toBe(4)

    // Verify there are 2 visible inputs (2 have been hidden in config)
    expect(visibleInputCount).toBe(2)
  })

  test('Reconnects inputs after configuration changed via manage dialog save', async ({
    comfyPage
  }) => {
    const expectSingleNode = async (type: string) => {
      const nodes = await comfyPage.nodeOps.getNodeRefsByType(type)
      expect(nodes).toHaveLength(1)
      return nodes[0]
    }
    const latent = await expectSingleNode('EmptyLatentImage')
    const sampler = await expectSingleNode('KSampler')
    // Remove existing link
    const samplerInput = await sampler.getInput(0)
    await samplerInput.removeLinks()
    // Group latent + sampler
    await latent.click('title', {
      modifiers: ['Shift']
    })
    await sampler.click('title', {
      modifiers: ['Shift']
    })
    const groupNode = await sampler.convertToGroupNode()
    // Connect node to group
    const ckpt = await expectSingleNode('CheckpointLoaderSimple')
    const input = await ckpt.connectOutput(0, groupNode, 0)
    expect(await input.getLinkCount()).toBe(1)
    // Modify the group node via manage dialog
    const manage = await groupNode.manageGroupNode()
    await manage.selectNode('KSampler')
    await manage.changeTab('Inputs')
    await manage.setLabel('model', 'test')
    await manage.save()
    await manage.close()
    // Ensure the link is still present
    expect(await input.getLinkCount()).toBe(1)
  })

  test('Loads from a workflow using the legacy path separator ("/")', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('groupnodes/legacy_group_node')
    expect(await comfyPage.nodeOps.getGraphNodesCount()).toBe(1)
    await expect(
      comfyPage.page.locator('.comfy-missing-nodes')
    ).not.toBeVisible()
  })

  test.describe('Copy and paste', () => {
    let groupNode: NodeReference | null
    const WORKFLOW_NAME = 'groupnodes/group_node_v1.3.3'
    const GROUP_NODE_CATEGORY = 'group nodes>workflow'
    const GROUP_NODE_PREFIX = 'workflow>'
    const GROUP_NODE_NAME = 'group_node' // Node name in given workflow
    const GROUP_NODE_TYPE = `${GROUP_NODE_PREFIX}${GROUP_NODE_NAME}`

    const isRegisteredLitegraph = async (comfyPage: ComfyPage) => {
      return await comfyPage.page.evaluate((nodeType: string) => {
        return !!window.LiteGraph!.registered_node_types[nodeType]
      }, GROUP_NODE_TYPE)
    }

    const isRegisteredNodeDefStore = async (comfyPage: ComfyPage) => {
      await comfyPage.menu.nodeLibraryTab.open()
      const groupNodesFolderCt = await comfyPage.menu.nodeLibraryTab
        .getFolder(GROUP_NODE_CATEGORY)
        .count()
      return groupNodesFolderCt === 1
    }

    const verifyNodeLoaded = async (
      comfyPage: ComfyPage,
      expectedCount: number
    ) => {
      expect(
        await comfyPage.nodeOps.getNodeRefsByType(GROUP_NODE_TYPE)
      ).toHaveLength(expectedCount)
      expect(await isRegisteredLitegraph(comfyPage)).toBe(true)
      expect(await isRegisteredNodeDefStore(comfyPage)).toBe(true)
    }

    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
      await comfyPage.workflow.loadWorkflow(WORKFLOW_NAME)
      groupNode = await comfyPage.nodeOps.getFirstNodeRef()
      if (!groupNode)
        throw new Error(`Group node not found in workflow ${WORKFLOW_NAME}`)
      await groupNode.copy()
    })

    test('Copies and pastes group node within the same workflow', async ({
      comfyPage
    }) => {
      await comfyPage.clipboard.paste()
      await verifyNodeLoaded(comfyPage, 2)
    })

    test('Copies and pastes group node after clearing workflow', async ({
      comfyPage
    }) => {
      // Set setting
      await comfyPage.settings.setSetting('Comfy.ConfirmClear', false)

      // Clear workflow
      await comfyPage.command.executeCommand('Comfy.ClearWorkflow')

      await comfyPage.clipboard.paste()
      await verifyNodeLoaded(comfyPage, 1)
    })

    test('Copies and pastes group node into a newly created blank workflow', async ({
      comfyPage
    }) => {
      await comfyPage.menu.topbar.triggerTopbarCommand(['New'])
      await comfyPage.clipboard.paste()
      await verifyNodeLoaded(comfyPage, 1)
    })

    test('Copies and pastes group node across different workflows', async ({
      comfyPage
    }) => {
      await comfyPage.workflow.loadWorkflow('default')
      await comfyPage.clipboard.paste()
      await verifyNodeLoaded(comfyPage, 1)
    })

    test('Serializes group node after copy and paste across workflows', async ({
      comfyPage
    }) => {
      await comfyPage.menu.topbar.triggerTopbarCommand(['New'])
      await comfyPage.clipboard.paste()
      const currentGraphState = await comfyPage.page.evaluate(() =>
        window.app!.graph!.serialize()
      )

      await test.step('Load workflow containing a group node pasted from a different workflow', async () => {
        await comfyPage.page.evaluate(
          (workflow) =>
            window.app!.loadGraphData(workflow as ComfyWorkflowJSON),
          currentGraphState
        )
        await comfyPage.nextFrame()
        await verifyNodeLoaded(comfyPage, 1)
      })
    })
  })

  test.describe('Keybindings', () => {
    test('Convert to group node, no selection', async ({ comfyPage }) => {
      await expect.poll(() => comfyPage.toast.getVisibleToastCount()).toBe(0)
      await comfyPage.page.keyboard.press('Alt+g')
      await expect.poll(() => comfyPage.toast.getVisibleToastCount()).toBe(1)
    })
    test('Convert to group node, selected 1 node', async ({ comfyPage }) => {
      await expect.poll(() => comfyPage.toast.getVisibleToastCount()).toBe(0)
      await comfyPage.canvas.click({
        position: DefaultGraphPositions.textEncodeNode1
      })
      await comfyPage.nextFrame()
      await comfyPage.page.keyboard.press('Alt+g')
      await expect.poll(() => comfyPage.toast.getVisibleToastCount()).toBe(1)
    })
  })
})
