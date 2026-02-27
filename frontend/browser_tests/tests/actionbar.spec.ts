import type { Response } from '@playwright/test'
import { expect, mergeTests } from '@playwright/test'

import type { StatusWsMessage } from '../../src/schemas/apiSchema'
import { comfyPageFixture } from '../fixtures/ComfyPage'
import { webSocketFixture } from '../fixtures/ws'
import type { WorkspaceStore } from '../types/globals'

const test = mergeTests(comfyPageFixture, webSocketFixture)

test.describe('Actionbar', { tag: '@ui' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
  })

  /**
   * This test ensures that the autoqueue change mode can only queue one change at a time
   */
  test('Does not auto-queue multiple changes at a time', async ({
    comfyPage,
    ws
  }) => {
    // Enable change auto-queue mode
    const queueOpts = await comfyPage.actionbar.queueButton.toggleOptions()
    expect(await queueOpts.getMode()).toBe('disabled')
    await queueOpts.setMode('change')
    await comfyPage.nextFrame()
    expect(await queueOpts.getMode()).toBe('change')
    await comfyPage.actionbar.queueButton.toggleOptions()

    // Intercept the prompt queue endpoint
    let promptNumber = 0
    await comfyPage.page.route('**/api/prompt', async (route, req) => {
      await new Promise((r) => setTimeout(r, 100))
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          prompt_id: promptNumber,
          number: ++promptNumber,
          node_errors: {},
          // Include the request data to validate which prompt was queued so we can validate the width
          __request: req.postDataJSON()
        })
      })
    })

    // Start watching for a message to prompt
    const requestPromise = comfyPage.page.waitForResponse('**/api/prompt')

    // Find and set the width on the latent node
    const triggerChange = async (value: number) => {
      return await comfyPage.page.evaluate((value) => {
        const node = window.app!.graph!._nodes.find(
          (n) => n.type === 'EmptyLatentImage'
        )
        node!.widgets![0].value = value

        ;(
          window.app!.extensionManager as WorkspaceStore
        ).workflow.activeWorkflow?.changeTracker.checkState()
      }, value)
    }

    // Trigger a status websocket message
    const triggerStatus = async (queueSize: number) => {
      await ws.trigger({
        type: 'status',
        data: {
          status: {
            exec_info: {
              queue_remaining: queueSize
            }
          }
        }
      } as StatusWsMessage)
    }

    // Extract the width from the queue response
    const getQueuedWidth = async (resp: Promise<Response>) => {
      const obj = await (await resp).json()
      return obj['__request']['prompt']['5']['inputs']['width']
    }

    // Trigger a bunch of changes
    const START = 32
    const END = 64
    for (let i = START; i <= END; i += 8) {
      await triggerChange(i)
    }

    // Ensure the queued width is the first value
    expect(
      await getQueuedWidth(requestPromise),
      'the first queued prompt should be the first change width'
    ).toBe(START)

    // Ensure that no other changes are queued
    await expect(
      comfyPage.page.waitForResponse('**/api/prompt', { timeout: 250 })
    ).rejects.toThrow()
    expect(
      promptNumber,
      'only 1 prompt should have been queued even though there were multiple changes'
    ).toBe(1)

    // Trigger a status update so auto-queue re-runs
    await triggerStatus(1)
    await triggerStatus(0)

    // Ensure the queued width is the last queued value
    expect(
      await getQueuedWidth(comfyPage.page.waitForResponse('**/api/prompt')),
      'last queued prompt width should be the last change'
    ).toBe(END)
    expect(promptNumber, 'queued prompt count should be 2').toBe(2)
  })

  test('Can dock actionbar into top menu', async ({ comfyPage }) => {
    await comfyPage.page.dragAndDrop(
      '.actionbar .drag-handle',
      '.actionbar-container',
      {
        targetPosition: { x: 50, y: 20 },
        force: true
      }
    )
    expect(await comfyPage.actionbar.isDocked()).toBe(true)
  })
})
