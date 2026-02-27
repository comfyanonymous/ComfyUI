import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../../fixtures/ComfyPage'

test.describe('Properties panel', () => {
  test('opens and updates title based on selection', async ({ comfyPage }) => {
    await comfyPage.actionbar.propertiesButton.click()

    const { propertiesPanel } = comfyPage.menu

    await expect(propertiesPanel.panelTitle).toContainText('Workflow Overview')

    await comfyPage.nodeOps.selectNodes([
      'KSampler',
      'CLIP Text Encode (Prompt)'
    ])

    await expect(propertiesPanel.panelTitle).toContainText('3 items selected')
    await expect(propertiesPanel.root.getByText('KSampler')).toHaveCount(1)
    await expect(
      propertiesPanel.root.getByText('CLIP Text Encode (Prompt)')
    ).toHaveCount(2)

    await propertiesPanel.searchBox.fill('seed')
    await expect(propertiesPanel.root.getByText('KSampler')).toHaveCount(1)
    await expect(
      propertiesPanel.root.getByText('CLIP Text Encode (Prompt)')
    ).toHaveCount(0)

    await propertiesPanel.searchBox.fill('')
    await expect(propertiesPanel.root.getByText('KSampler')).toHaveCount(1)
    await expect(
      propertiesPanel.root.getByText('CLIP Text Encode (Prompt)')
    ).toHaveCount(2)
  })
})
