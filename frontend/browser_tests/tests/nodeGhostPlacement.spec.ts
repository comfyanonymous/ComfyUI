import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../fixtures/ComfyPage'

type ComfyPage = Parameters<Parameters<typeof test>[2]>[0]['comfyPage']

async function setVueMode(comfyPage: ComfyPage, enabled: boolean) {
  await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', enabled)
  if (enabled) {
    await comfyPage.vueNodes.waitForNodes()
  }
}

async function addGhostAtCenter(comfyPage: ComfyPage) {
  await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')
  await comfyPage.nextFrame()

  const viewport = comfyPage.page.viewportSize()!
  const centerX = Math.round(viewport.width / 2)
  const centerY = Math.round(viewport.height / 2)

  await comfyPage.page.mouse.move(centerX, centerY)
  await comfyPage.nextFrame()

  const nodeId = await comfyPage.page.evaluate(
    ([clientX, clientY]) => {
      const node = window.LiteGraph!.createNode('VAEDecode')!
      const event = new MouseEvent('click', { clientX, clientY })
      window.app!.graph.add(node, { ghost: true, dragEvent: event })
      return node.id
    },
    [centerX, centerY] as const
  )
  await comfyPage.nextFrame()

  return { nodeId, centerX, centerY }
}

function getNodeById(comfyPage: ComfyPage, nodeId: number | string) {
  return comfyPage.page.evaluate((id) => {
    const node = window.app!.graph.getNodeById(id)
    if (!node) return null
    return { ghost: !!node.flags.ghost }
  }, nodeId)
}

for (const mode of ['litegraph', 'vue'] as const) {
  test.describe(`Ghost node placement (${mode} mode)`, () => {
    test.beforeEach(async ({ comfyPage }) => {
      await setVueMode(comfyPage, mode === 'vue')
    })

    test('positions ghost node at cursor', async ({ comfyPage }) => {
      await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')
      await comfyPage.nextFrame()

      const viewport = comfyPage.page.viewportSize()!
      const centerX = Math.round(viewport.width / 2)
      const centerY = Math.round(viewport.height / 2)

      await comfyPage.page.mouse.move(centerX, centerY)
      await comfyPage.nextFrame()

      const result = await comfyPage.page.evaluate(
        ([clientX, clientY]) => {
          const node = window.LiteGraph!.createNode('VAEDecode')!
          const event = new MouseEvent('click', { clientX, clientY })
          window.app!.graph.add(node, { ghost: true, dragEvent: event })

          const canvas = window.app!.canvas
          const rect = canvas.canvas.getBoundingClientRect()
          const cursorCanvasX =
            (clientX - rect.left) / canvas.ds.scale - canvas.ds.offset[0]
          const cursorCanvasY =
            (clientY - rect.top) / canvas.ds.scale - canvas.ds.offset[1]

          return {
            diffX: node.pos[0] + node.size[0] / 2 - cursorCanvasX,
            diffY: node.pos[1] - 10 - cursorCanvasY
          }
        },
        [centerX, centerY] as const
      )
      await comfyPage.nextFrame()

      expect(Math.abs(result.diffX)).toBeLessThan(5)
      expect(Math.abs(result.diffY)).toBeLessThan(5)
    })

    test('left-click confirms ghost placement', async ({ comfyPage }) => {
      const { nodeId, centerX, centerY } = await addGhostAtCenter(comfyPage)

      const before = await getNodeById(comfyPage, nodeId)
      expect(before).not.toBeNull()
      expect(before!.ghost).toBe(true)

      await comfyPage.page.mouse.click(centerX, centerY)
      await comfyPage.nextFrame()

      const after = await getNodeById(comfyPage, nodeId)
      expect(after).not.toBeNull()
      expect(after!.ghost).toBe(false)
    })

    test('Escape cancels ghost placement', async ({ comfyPage }) => {
      const { nodeId } = await addGhostAtCenter(comfyPage)

      const before = await getNodeById(comfyPage, nodeId)
      expect(before).not.toBeNull()
      expect(before!.ghost).toBe(true)

      await comfyPage.page.keyboard.press('Escape')
      await comfyPage.nextFrame()

      const after = await getNodeById(comfyPage, nodeId)
      expect(after).toBeNull()
    })

    test('Delete cancels ghost placement', async ({ comfyPage }) => {
      const { nodeId } = await addGhostAtCenter(comfyPage)

      const before = await getNodeById(comfyPage, nodeId)
      expect(before).not.toBeNull()
      expect(before!.ghost).toBe(true)

      await comfyPage.page.keyboard.press('Delete')
      await comfyPage.nextFrame()

      const after = await getNodeById(comfyPage, nodeId)
      expect(after).toBeNull()
    })

    test('Backspace cancels ghost placement', async ({ comfyPage }) => {
      const { nodeId } = await addGhostAtCenter(comfyPage)

      const before = await getNodeById(comfyPage, nodeId)
      expect(before).not.toBeNull()
      expect(before!.ghost).toBe(true)

      await comfyPage.page.keyboard.press('Backspace')
      await comfyPage.nextFrame()

      const after = await getNodeById(comfyPage, nodeId)
      expect(after).toBeNull()
    })

    test('right-click cancels ghost placement', async ({ comfyPage }) => {
      const { nodeId, centerX, centerY } = await addGhostAtCenter(comfyPage)

      const before = await getNodeById(comfyPage, nodeId)
      expect(before).not.toBeNull()
      expect(before!.ghost).toBe(true)

      await comfyPage.page.mouse.click(centerX, centerY, { button: 'right' })
      await comfyPage.nextFrame()

      const after = await getNodeById(comfyPage, nodeId)
      expect(after).toBeNull()
    })
  })
}
