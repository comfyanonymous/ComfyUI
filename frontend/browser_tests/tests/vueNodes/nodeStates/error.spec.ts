import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../fixtures/ComfyPage'

const ERROR_CLASS = /border-node-stroke-error/

test.describe('Vue Node Error', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.vueNodes.waitForNodes()
  })

  test('should display error state when node is missing (node from workflow is not installed)', async ({
    comfyPage
  }) => {
    await comfyPage.setup()
    await comfyPage.workflow.loadWorkflow('missing/missing_nodes')

    // Expect error state on missing unknown node
    const unknownNode = comfyPage.page.locator('[data-node-id]').filter({
      hasText: 'UNKNOWN NODE'
    })
    await expect(unknownNode).toHaveClass(ERROR_CLASS)
  })

  test('should display error state when node causes execution error', async ({
    comfyPage
  }) => {
    await comfyPage.setup()
    await comfyPage.workflow.loadWorkflow('nodes/execution_error')
    await comfyPage.runButton.click()

    const raiseErrorNode = comfyPage.vueNodes.getNodeByTitle('Raise Error')
    await expect(raiseErrorNode).toHaveClass(ERROR_CLASS)
  })
})
