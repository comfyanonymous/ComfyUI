import type { Page } from '@playwright/test'
import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

async function checkTemplateFileExists(
  page: Page,
  filename: string
): Promise<boolean> {
  const response = await page.request.head(
    new URL(`/templates/${filename}`, page.url()).toString()
  )
  return response.ok()
}

test.describe('Templates', { tag: ['@slow', '@workflow'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting(
      'Comfy.Workflow.ShowMissingModelsWarning',
      false
    )
  })

  test('should have a JSON workflow file for each template', async ({
    comfyPage
  }) => {
    test.slow()
    const templates = await comfyPage.templates.getAllTemplates()
    for (const template of templates) {
      const exists = await checkTemplateFileExists(
        comfyPage.page,
        `${template.name}.json`
      )
      expect(exists, `Missing workflow: ${template.name}`).toBe(true)
    }
  })

  // TODO: Re-enable this test once issue resolved
  // https://github.com/Comfy-Org/ComfyUI_frontend/issues/3992
  test.skip('should have all required thumbnail media for each template', async ({
    comfyPage
  }) => {
    test.slow()
    const templates = await comfyPage.templates.getAllTemplates()
    for (const template of templates) {
      const { name, mediaSubtype, thumbnailVariant } = template
      const baseMedia = `${name}-1.${mediaSubtype}`

      // Check base thumbnail
      const baseExists = await checkTemplateFileExists(
        comfyPage.page,
        baseMedia
      )
      expect(baseExists, `Missing base thumbnail: ${baseMedia}`).toBe(true)

      // Check second thumbnail for variants that need it
      if (
        thumbnailVariant === 'compareSlider' ||
        thumbnailVariant === 'hoverDissolve'
      ) {
        const secondMedia = `${name}-2.${mediaSubtype}`
        const secondExists = await checkTemplateFileExists(
          comfyPage.page,
          secondMedia
        )
        expect(
          secondExists,
          `Missing second thumbnail: ${secondMedia} required for ${thumbnailVariant}`
        ).toBe(true)
      }
    }
  })

  test('Can load template workflows', async ({ comfyPage }) => {
    // Clear the workflow
    await comfyPage.menu.workflowsTab.open()
    await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')
    await expect(async () => {
      expect(await comfyPage.nodeOps.getGraphNodesCount()).toBe(0)
    }).toPass({ timeout: 250 })

    // Load a template
    await comfyPage.command.executeCommand('Comfy.BrowseTemplates')
    await expect(comfyPage.templates.content).toBeVisible()

    await comfyPage.page
      .getByRole('button', { name: 'Getting Started' })
      .click()
    await comfyPage.templates.loadTemplate('default')
    await expect(comfyPage.templates.content).toBeHidden()

    // Ensure we now have some nodes
    await expect(async () => {
      expect(await comfyPage.nodeOps.getGraphNodesCount()).toBeGreaterThan(0)
    }).toPass({ timeout: 250 })
  })

  test('dialog should be automatically shown to first-time users', async ({
    comfyPage
  }) => {
    // Set the tutorial as not completed to mark the user as a first-time user
    await comfyPage.settings.setSetting('Comfy.TutorialCompleted', false)

    // Load the page
    await comfyPage.setup({ clearStorage: true })

    // Expect the templates dialog to be shown
    expect(await comfyPage.templates.content.isVisible()).toBe(true)
  })

  test('Uses proper locale files for templates', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.Locale', 'fr')

    await comfyPage.command.executeCommand('Comfy.BrowseTemplates')

    const dialog = comfyPage.page.getByRole('dialog').filter({
      has: comfyPage.page.getByRole('heading', { name: 'Modèles', exact: true })
    })
    await expect(dialog).toBeVisible()

    // Validate that French-localized strings from the templates index are rendered
    await expect(
      dialog.getByRole('heading', { name: 'Modèles', exact: true })
    ).toBeVisible()
    await expect(
      dialog.getByRole('button', { name: 'Tous les modèles', exact: true })
    ).toBeVisible()

    // Ensure the English fallback copy is not shown anywhere
    await expect(
      comfyPage.page.getByText('All Templates', { exact: true })
    ).toHaveCount(0)
  })

  test('Falls back to English templates when locale file not found', async ({
    comfyPage
  }) => {
    // Set locale to a language that doesn't have a template file
    await comfyPage.settings.setSetting('Comfy.Locale', 'de') // German - no index.de.json exists

    // Wait for the German request (expected to 404)
    const germanRequestPromise = comfyPage.page.waitForRequest(
      '**/templates/index.de.json'
    )

    // Wait for the fallback English request
    const englishRequestPromise = comfyPage.page.waitForRequest(
      '**/templates/index.json'
    )

    // Intercept the German file to simulate a 404
    await comfyPage.page.route('**/templates/index.de.json', async (route) => {
      await route.fulfill({
        status: 404,
        headers: { 'Content-Type': 'text/plain' },
        body: 'Not Found'
      })
    })

    // Allow the English index to load normally
    await comfyPage.page.route('**/templates/index.json', (route) =>
      route.continue()
    )

    // Load the templates dialog
    await comfyPage.command.executeCommand('Comfy.BrowseTemplates')
    await expect(comfyPage.templates.content).toBeVisible()

    // Verify German was requested first, then English as fallback
    const germanRequest = await germanRequestPromise
    const englishRequest = await englishRequestPromise

    expect(germanRequest.url()).toContain('templates/index.de.json')
    expect(englishRequest.url()).toContain('templates/index.json')

    // Verify English titles are shown as fallback
    await expect(
      comfyPage.page.getByRole('main').getByText('All Templates')
    ).toBeVisible()
  })

  test('template cards are dynamically sized and responsive', async ({
    comfyPage
  }) => {
    // Open templates dialog
    await comfyPage.command.executeCommand('Comfy.BrowseTemplates')
    await comfyPage.templates.content.waitFor({ state: 'visible' })

    const templateGrid = comfyPage.page.locator(
      '[data-testid="template-workflows-content"]'
    )
    const nav = comfyPage.page.locator('header', { hasText: 'Templates' })

    await comfyPage.templates.expectMinimumCardCount(1)
    await expect(templateGrid).toBeVisible()
    await expect(nav).toBeVisible() // Nav should be visible at desktop size

    const mobileSize = { width: 640, height: 800 }
    await comfyPage.page.setViewportSize(mobileSize)
    await comfyPage.templates.expectMinimumCardCount(1)
    await expect(templateGrid).toBeVisible()
    // Nav header is clipped by overflow-hidden parent at mobile size
    await expect(nav).not.toBeInViewport()

    const tabletSize = { width: 1024, height: 800 }
    await comfyPage.page.setViewportSize(tabletSize)
    await comfyPage.templates.expectMinimumCardCount(1)
    await expect(templateGrid).toBeVisible()
    await expect(nav).toBeVisible() // Nav should be visible at tablet size
  })

  test(
    'template cards descriptions adjust height dynamically',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      // Setup test by intercepting templates response to inject cards with varying description lengths
      await comfyPage.page.route(
        '**/templates/index.json',
        async (route, _) => {
          const response = [
            {
              moduleName: 'default',
              title: 'Test Templates',
              type: 'image',
              templates: [
                {
                  name: 'short-description',
                  title: 'Short Description',
                  mediaType: 'image',
                  mediaSubtype: 'webp',
                  description: 'This is a short description.'
                },
                {
                  name: 'medium-description',
                  title: 'Medium Description',
                  mediaType: 'image',
                  mediaSubtype: 'webp',
                  description:
                    'This is a medium length description that should take up two lines on most displays.'
                },
                {
                  name: 'long-description',
                  title: 'Long Description',
                  mediaType: 'image',
                  mediaSubtype: 'webp',
                  description:
                    'This is a much longer description that should definitely wrap to multiple lines. It contains enough text to demonstrate how the cards handle varying amounts of content while maintaining a consistent layout grid.'
                }
              ]
            }
          ]
          await route.fulfill({
            status: 200,
            body: JSON.stringify(response),
            headers: {
              'Content-Type': 'application/json',
              'Cache-Control': 'no-store'
            }
          })
        }
      )

      // Mock the thumbnail images to avoid 404s
      await comfyPage.page.route('**/templates/**.webp', async (route) => {
        const headers = {
          'Content-Type': 'image/webp',
          'Cache-Control': 'no-store'
        }
        await route.fulfill({
          status: 200,
          path: 'browser_tests/assets/example.webp',
          headers
        })
      })

      // Open templates dialog
      await comfyPage.command.executeCommand('Comfy.BrowseTemplates')
      await expect(comfyPage.templates.content).toBeVisible()

      // Wait for cards to load
      await expect(
        comfyPage.page.locator(
          '[data-testid="template-workflow-short-description"]'
        )
      ).toBeVisible({ timeout: 5000 })

      // Verify all three cards with different descriptions are visible
      const shortDescCard = comfyPage.page.locator(
        '[data-testid="template-workflow-short-description"]'
      )
      const mediumDescCard = comfyPage.page.locator(
        '[data-testid="template-workflow-medium-description"]'
      )
      const longDescCard = comfyPage.page.locator(
        '[data-testid="template-workflow-long-description"]'
      )

      await expect(shortDescCard).toBeVisible()
      await expect(mediumDescCard).toBeVisible()
      await expect(longDescCard).toBeVisible()

      // Verify descriptions are visible and have line-clamp class
      // The description is in a p tag with text-muted class
      const shortDesc = shortDescCard.locator('p.text-muted.line-clamp-2')
      const mediumDesc = mediumDescCard.locator('p.text-muted.line-clamp-2')
      const longDesc = longDescCard.locator('p.text-muted.line-clamp-2')

      await expect(shortDesc).toContainText('short description')
      await expect(mediumDesc).toContainText('medium length description')
      await expect(longDesc).toContainText('much longer description')

      // Verify grid layout maintains consistency
      const templateGrid = comfyPage.page.locator(
        '[data-testid="template-workflows-content"]'
      )
      await expect(templateGrid).toBeVisible()
      await expect(templateGrid).toHaveScreenshot(
        'template-grid-varying-content.png'
      )
    }
  )
})
