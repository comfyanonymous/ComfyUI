import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Subgraph duplicate ID remapping', { tag: ['@subgraph'] }, () => {
  const WORKFLOW = 'subgraphs/subgraph-nested-duplicate-ids'

  test('All node IDs are globally unique after loading', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(WORKFLOW)

    const result = await comfyPage.page.evaluate(() => {
      const graph = window.app!.canvas.graph!
      // TODO: Extract allGraphs accessor (root + subgraphs) into LGraph
      // TODO: Extract allNodeIds accessor into LGraph
      const allGraphs = [graph, ...graph.subgraphs.values()]
      const allIds = allGraphs
        .flatMap((g) => g._nodes)
        .map((n) => n.id)
        .filter((id): id is number => typeof id === 'number')

      return { allIds, uniqueCount: new Set(allIds).size }
    })

    expect(result.uniqueCount).toBe(result.allIds.length)
    expect(result.allIds.length).toBeGreaterThanOrEqual(10)
  })

  test('Root graph node IDs are preserved as canonical', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(WORKFLOW)

    const rootIds = await comfyPage.page.evaluate(() => {
      const graph = window.app!.canvas.graph!
      return graph._nodes
        .map((n) => n.id)
        .filter((id): id is number => typeof id === 'number')
        .sort((a, b) => a - b)
    })

    expect(rootIds).toEqual([1, 2, 5])
  })

  test('All links reference valid nodes in their graph', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(WORKFLOW)

    const invalidLinks = await comfyPage.page.evaluate(() => {
      const graph = window.app!.canvas.graph!
      const labeledGraphs: [string, typeof graph][] = [
        ['root', graph],
        ...[...graph.subgraphs.entries()].map(
          ([id, sg]) => [`subgraph:${id}`, sg] as [string, typeof graph]
        )
      ]

      const isNonNegative = (id: number | string) =>
        typeof id === 'number' && id >= 0

      return labeledGraphs.flatMap(([label, g]) =>
        [...g._links.values()].flatMap((link) =>
          [
            isNonNegative(link.origin_id) &&
              !g._nodes_by_id[link.origin_id] &&
              `${label}: origin_id ${link.origin_id} not found`,
            isNonNegative(link.target_id) &&
              !g._nodes_by_id[link.target_id] &&
              `${label}: target_id ${link.target_id} not found`
          ].filter(Boolean)
        )
      )
    })

    expect(invalidLinks).toEqual([])
  })

  test('Subgraph navigation works after ID remapping', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(WORKFLOW)

    const subgraphNode = await comfyPage.nodeOps.getNodeRefById('5')
    await subgraphNode.navigateIntoSubgraph()

    const isInSubgraph = () =>
      comfyPage.page.evaluate(
        () => window.app!.canvas.graph?.isRootGraph === false
      )

    expect(await isInSubgraph()).toBe(true)

    await comfyPage.page.keyboard.press('Escape')
    await comfyPage.nextFrame()

    expect(await isInSubgraph()).toBe(false)
  })
})
