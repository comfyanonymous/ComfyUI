import { expect } from '@playwright/test'

import type { ComfyPage } from '../fixtures/ComfyPage'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'

async function createSubgraphAndNavigateInto(comfyPage: ComfyPage) {
  await comfyPage.workflow.loadWorkflow('default')
  await comfyPage.nextFrame()

  const ksampler = await comfyPage.nodeOps.getNodeRefById('3')
  await ksampler.click('title')
  await ksampler.convertToSubgraph()
  await comfyPage.nextFrame()

  const subgraphNodes =
    await comfyPage.nodeOps.getNodeRefsByTitle('New Subgraph')
  expect(subgraphNodes.length).toBe(1)
  const subgraphNode = subgraphNodes[0]

  await subgraphNode.navigateIntoSubgraph()
  return subgraphNode
}

async function exitSubgraphAndPublish(
  comfyPage: ComfyPage,
  subgraphNode: Awaited<ReturnType<typeof createSubgraphAndNavigateInto>>,
  blueprintName: string
) {
  await comfyPage.page.keyboard.press('Escape')
  await comfyPage.nextFrame()

  await subgraphNode.click('title')
  await comfyPage.command.executeCommand('Comfy.PublishSubgraph', {
    name: blueprintName
  })

  await expect(comfyPage.visibleToasts).toHaveCount(1, { timeout: 5000 })
  await comfyPage.toast.closeToasts(1)
}

async function searchAndExpectResult(
  comfyPage: ComfyPage,
  searchTerm: string,
  expectedResult: string
) {
  await comfyPage.command.executeCommand('Workspace.SearchBox.Toggle')
  await expect(comfyPage.searchBox.input).toHaveCount(1)
  await comfyPage.searchBox.input.fill(searchTerm)
  await expect(comfyPage.searchBox.findResult(expectedResult)).toBeVisible({
    timeout: 10000
  })
}

test.describe('Subgraph Search Aliases', { tag: ['@subgraph'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )
  })

  test('Can set search aliases on subgraph and find via search', async ({
    comfyPage
  }) => {
    const subgraphNode = await createSubgraphAndNavigateInto(comfyPage)

    await comfyPage.command.executeCommand('Comfy.Subgraph.SetSearchAliases', {
      aliases: 'qwerty,unicorn'
    })

    const blueprintName = `test-aliases-${Date.now()}`
    await exitSubgraphAndPublish(comfyPage, subgraphNode, blueprintName)
    await searchAndExpectResult(comfyPage, 'unicorn', blueprintName)
  })

  test('Can set description on subgraph', async ({ comfyPage }) => {
    await createSubgraphAndNavigateInto(comfyPage)

    await comfyPage.command.executeCommand('Comfy.Subgraph.SetDescription', {
      description: 'This is a test description'
    })
    // Verify the description was set on the subgraph's extra
    const description = await comfyPage.page.evaluate(() => {
      const subgraph = window['app']!.canvas.subgraph
      return (subgraph?.extra as Record<string, unknown>)?.BlueprintDescription
    })
    expect(description).toBe('This is a test description')
  })

  test('Search aliases persist after publish and reload', async ({
    comfyPage
  }) => {
    const subgraphNode = await createSubgraphAndNavigateInto(comfyPage)

    await comfyPage.command.executeCommand('Comfy.Subgraph.SetSearchAliases', {
      aliases: 'dragon, fire breather'
    })

    const blueprintName = `test-persist-${Date.now()}`
    await exitSubgraphAndPublish(comfyPage, subgraphNode, blueprintName)

    // Reload the page to ensure aliases are persisted
    await comfyPage.page.reload()
    await comfyPage.page.waitForFunction(
      () => window['app'] && window['app'].extensionManager
    )
    await comfyPage.nextFrame()

    await searchAndExpectResult(comfyPage, 'dragon', blueprintName)
  })
})
