import { expect } from '@playwright/test'

import type { SystemStats } from '../../src/schemas/apiSchema'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Version Mismatch Warnings', { tag: '@slow' }, () => {
  const ALWAYS_AHEAD_OF_INSTALLED_VERSION = '100.100.100'
  const ALWAYS_BEHIND_INSTALLED_VERSION = '0.0.0'

  const createMockSystemStatsRes = (
    requiredFrontendVersion: string
  ): SystemStats => {
    return {
      system: {
        os: 'posix',
        ram_total: 67235385344,
        ram_free: 13464207360,
        comfyui_version: '0.3.46',
        required_frontend_version: requiredFrontendVersion,
        python_version: '3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0]',
        pytorch_version: '2.6.0+cu124',
        embedded_python: false,
        argv: ['main.py']
      },
      devices: [
        {
          name: 'cuda:0 NVIDIA GeForce RTX 4070 : cudaMallocAsync',
          type: 'cuda',
          index: 0,
          vram_total: 12557156352,
          vram_free: 2439249920,
          torch_vram_total: 0,
          torch_vram_free: 0
        }
      ]
    }
  }

  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting(
      'Comfy.VersionCompatibility.DisableWarnings',
      false
    )
  })

  test('should show version mismatch warnings when installed version lower than required', async ({
    comfyPage
  }) => {
    // Mock system_stats route to indicate that the installed version is always ahead of the required version
    await comfyPage.page.route('**/system_stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          createMockSystemStatsRes(ALWAYS_AHEAD_OF_INSTALLED_VERSION)
        )
      })
    })
    await comfyPage.setup()

    // Expect a warning toast to be shown
    await expect(
      comfyPage.page.getByText('Version Compatibility Warning')
    ).toBeVisible()
  })

  test('should not show version mismatch warnings when installed version is ahead of required', async ({
    comfyPage
  }) => {
    // Mock system_stats route to indicate that the installed version is always ahead of the required version
    await comfyPage.page.route('**/system_stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          createMockSystemStatsRes(ALWAYS_BEHIND_INSTALLED_VERSION)
        )
      })
    })
    await comfyPage.setup()

    // Expect no warning toast to be shown
    await expect(
      comfyPage.page.getByText('Version Compatibility Warning')
    ).not.toBeVisible()
  })

  test('should persist dismissed state across sessions', async ({
    comfyPage
  }) => {
    test.setTimeout(30_000)
    // Mock system_stats route to indicate that the installed version is always ahead of the required version
    await comfyPage.page.route('**/system_stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          createMockSystemStatsRes(ALWAYS_AHEAD_OF_INSTALLED_VERSION)
        )
      })
    })
    await comfyPage.setup()

    // Locate the warning toast and dismiss it
    const warningToast = comfyPage.page.locator('.p-toast-message').filter({
      hasText: 'Version Compatibility'
    })
    await warningToast.waitFor({ state: 'visible' })
    const dismissButton = warningToast.getByRole('button', { name: 'Close' })
    await dismissButton.click()

    // Wait for the dismissed state to be persisted
    await comfyPage.page.waitForFunction(
      () => !!localStorage.getItem('comfy.versionMismatch.dismissals')
    )

    // Reload the page, keeping local storage
    await comfyPage.setup({ clearStorage: false })

    // The same warning from same versions should not be shown to the user again
    await expect(
      comfyPage.page.getByText('Version Compatibility Warning')
    ).not.toBeVisible()
  })
})
