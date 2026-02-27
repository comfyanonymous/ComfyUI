import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../fixtures/ComfyPage'
import type { ComfyPage } from '../fixtures/ComfyPage'
import { fitToViewInstant } from '../helpers/fitToView'
import type { NodeReference } from '../fixtures/utils/litegraphUtils'

// TODO: there might be a better solution for this
// Helper function to pan canvas and select node
async function selectNodeWithPan(comfyPage: ComfyPage, nodeRef: NodeReference) {
  const nodePos = await nodeRef.getPosition()

  await comfyPage.page.evaluate((pos) => {
    const app = window.app!
    const canvas = app.canvas
    canvas.ds.offset[0] = -pos.x + canvas.canvas.width / 2
    canvas.ds.offset[1] = -pos.y + canvas.canvas.height / 2 + 100
    canvas.setDirty(true, true)
  }, nodePos)

  await comfyPage.nextFrame()
  await nodeRef.click('title')
}

test.describe('Node Help', { tag: ['@slow', '@ui'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.setup()
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.NewDesign', false)
  })

  test.describe('Selection Toolbox', () => {
    test('Should open help menu for selected node', async ({ comfyPage }) => {
      // Load a workflow with a node
      await comfyPage.settings.setSetting('Comfy.Canvas.SelectionToolbox', true)
      await comfyPage.workflow.loadWorkflow('default')

      // Select a single node (KSampler) using node references
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      if (ksamplerNodes.length === 0) {
        throw new Error('No KSampler nodes found in the workflow')
      }

      // Select the node with panning to ensure toolbox is visible
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      // Wait for selection toolbox to appear
      await expect(comfyPage.selectionToolbox).toBeVisible()

      // Click the help button in the selection toolbox
      const helpButton = comfyPage.selectionToolbox.locator(
        'button[data-testid="info-button"]'
      )
      await expect(helpButton).toBeVisible()
      await helpButton.click()

      // Verify that the help page is shown for the correct node
      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )
      await expect(helpPage).toContainText('KSampler')
      await expect(helpPage.locator('.node-help-content')).toBeVisible()
    })
  })

  test.describe('Node Library Sidebar', () => {
    test('Should open help menu from node library', async ({ comfyPage }) => {
      // Open the node library sidebar
      await comfyPage.menu.nodeLibraryTab.open()

      // Wait for node library to load
      await expect(comfyPage.menu.nodeLibraryTab.nodeLibraryTree).toBeVisible()

      // Search for KSampler to make it easier to find
      await comfyPage.menu.nodeLibraryTab.nodeLibrarySearchBoxInput.fill(
        'KSampler'
      )

      // Find the KSampler node in search results
      const ksamplerNode = comfyPage.page
        .locator('.tree-explorer-node-label')
        .filter({ hasText: 'KSampler' })
        .first()
      await expect(ksamplerNode).toBeVisible()

      // Hover over the node to show action buttons
      await ksamplerNode.hover()

      // Click the help button
      const helpButton = ksamplerNode.getByRole('button', {
        name: /learn more/i
      })
      await expect(helpButton).toBeVisible()
      await helpButton.click()

      // Verify that the help page is shown
      const helpPage = comfyPage.page.locator('.sidebar-content-container')
      await expect(helpPage).toContainText('KSampler')
      await expect(helpPage.locator('.node-help-content')).toBeVisible()
    })

    test('Should show node library tab when clicking back from help page', async ({
      comfyPage
    }) => {
      // Open the node library sidebar
      await comfyPage.menu.nodeLibraryTab.open()

      // Wait for node library to load
      await expect(comfyPage.menu.nodeLibraryTab.nodeLibraryTree).toBeVisible()

      // Search for KSampler
      await comfyPage.menu.nodeLibraryTab.nodeLibrarySearchBoxInput.fill(
        'KSampler'
      )

      // Find and interact with the node
      const ksamplerNode = comfyPage.page
        .locator('.tree-explorer-node-label')
        .filter({ hasText: 'KSampler' })
        .first()
      await ksamplerNode.hover()
      const helpButton = ksamplerNode.getByRole('button', {
        name: /learn more/i
      })
      await helpButton.click()

      // Verify help page is shown
      const helpPage = comfyPage.page.locator('.sidebar-content-container')
      await expect(helpPage).toContainText('KSampler')

      // Click the back button - use a more specific selector
      const backButton = helpPage.getByRole('button', { name: /back/i })
      await backButton.click()

      // Verify that we're back to the node library view
      await expect(comfyPage.menu.nodeLibraryTab.nodeLibraryTree).toBeVisible()
      await expect(
        comfyPage.menu.nodeLibraryTab.nodeLibrarySearchBoxInput
      ).toBeVisible()

      // Verify help page is no longer visible
      await expect(helpPage.locator('.node-help-content')).not.toBeVisible()
    })
  })

  test.describe('Help Content', () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.Canvas.SelectionToolbox', true)
    })

    test('Should display loading state while fetching help', async ({
      comfyPage
    }) => {
      // Mock slow network response
      await comfyPage.page.route('**/docs/**/*.md', async (route) => {
        await new Promise((resolve) => setTimeout(resolve, 1000))
        await route.fulfill({
          status: 200,
          body: '# Test Help Content\nThis is test help content.'
        })
      })

      // Load workflow and select a node
      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      // Click help button
      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      // Verify loading spinner is shown
      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )
      await expect(helpPage.locator('.p-progressspinner')).toBeVisible()

      // Wait for content to load
      await expect(helpPage).toContainText('Test Help Content')
    })

    test('Should display fallback content when help file not found', async ({
      comfyPage
    }) => {
      // Mock 404 response for help files
      await comfyPage.page.route('**/docs/**/*.md', async (route) => {
        await route.fulfill({
          status: 404,
          body: 'Not Found'
        })
      })

      // Load workflow and select a node
      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      // Click help button
      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      // Verify fallback content is shown (description, inputs, outputs)
      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )
      await expect(helpPage).toContainText('Description')
      await expect(helpPage).toContainText('Inputs')
      await expect(helpPage).toContainText('Outputs')
    })

    test('Should render markdown with images correctly', async ({
      comfyPage
    }) => {
      // Mock response with markdown containing images
      await comfyPage.page.route('**/docs/KSampler/en.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: `# KSampler Documentation

![Example Image](example.jpg)
![External Image](https://example.com/image.png)

## Parameters
- **steps**: Number of steps
`
        })
      })

      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )
      await expect(helpPage).toContainText('KSampler Documentation')

      // Check that relative image paths are prefixed correctly
      const relativeImage = helpPage.locator('img[alt="Example Image"]')
      await expect(relativeImage).toBeVisible()
      await expect(relativeImage).toHaveAttribute(
        'src',
        /.*\/docs\/KSampler\/example\.jpg/
      )

      // Check that absolute URLs are not modified
      const externalImage = helpPage.locator('img[alt="External Image"]')
      await expect(externalImage).toHaveAttribute(
        'src',
        'https://example.com/image.png'
      )
    })

    test('Should render video elements with source tags in markdown', async ({
      comfyPage
    }) => {
      // Mock response with video elements
      await comfyPage.page.route('**/docs/KSampler/en.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: `# KSampler Demo

<video src="demo.mp4" controls autoplay></video>
<video src="/absolute/video.mp4" controls></video>

<video controls>
  <source src="video.mp4" type="video/mp4">
  <source src="https://example.com/video.webm" type="video/webm">
</video>
`
        })
      })

      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )

      // Check relative video paths are prefixed
      const relativeVideo = helpPage.locator('video[src*="demo.mp4"]')
      await expect(relativeVideo).toBeVisible()
      await expect(relativeVideo).toHaveAttribute(
        'src',
        /.*\/docs\/KSampler\/demo\.mp4/
      )
      await expect(relativeVideo).toHaveAttribute('controls', '')
      await expect(relativeVideo).toHaveAttribute('autoplay', '')

      // Check absolute paths are not modified
      const absoluteVideo = helpPage.locator('video[src="/absolute/video.mp4"]')
      await expect(absoluteVideo).toHaveAttribute('src', '/absolute/video.mp4')

      // Check video source elements
      const relativeVideoSource = helpPage.locator('source[src*="video.mp4"]')
      await expect(relativeVideoSource).toHaveAttribute(
        'src',
        /.*\/docs\/KSampler\/video\.mp4/
      )

      const externalVideoSource = helpPage.locator(
        'source[src="https://example.com/video.webm"]'
      )
      await expect(externalVideoSource).toHaveAttribute(
        'src',
        'https://example.com/video.webm'
      )
    })

    test('Should handle custom node documentation paths', async ({
      comfyPage
    }) => {
      // First load workflow with custom node
      await comfyPage.workflow.loadWorkflow('groupnodes/group_node_v1.3.3')

      // Mock custom node documentation with fallback
      await comfyPage.page.route(
        '**/extensions/*/docs/*/en.md',
        async (route) => {
          await route.fulfill({ status: 404 })
        }
      )

      await comfyPage.page.route('**/extensions/*/docs/*.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: `# Custom Node Documentation

This is documentation for a custom node.

![Custom Image](assets/custom.png)
`
        })
      })

      // Find and select a custom/group node
      const nodeRefs = await comfyPage.page.evaluate(() => {
        return window.app!.graph!.nodes.map((n) => n.id)
      })
      if (nodeRefs.length > 0) {
        const firstNode = await comfyPage.nodeOps.getNodeRefById(nodeRefs[0])
        await selectNodeWithPan(comfyPage, firstNode)
      }

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      if (await helpButton.isVisible()) {
        await helpButton.click()

        const helpPage = comfyPage.page.locator(
          '[data-testid="properties-panel"]'
        )
        await expect(helpPage).toContainText('Custom Node Documentation')

        // Check image path for custom nodes
        const image = helpPage.locator('img[alt="Custom Image"]')
        await expect(image).toHaveAttribute(
          'src',
          /.*\/extensions\/.*\/docs\/assets\/custom\.png/
        )
      }
    })

    test('Should sanitize dangerous HTML content', async ({ comfyPage }) => {
      // Mock response with potentially dangerous content
      await comfyPage.page.route('**/docs/KSampler/en.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: `# Safe Content

<script>alert('XSS')</script>
<img src="x" onerror="alert('XSS')">
<a href="javascript:alert('XSS')">Dangerous Link</a>
<iframe src="evil.com"></iframe>

<!-- Safe content -->
<video src="safe.mp4" controls></video>
<img src="safe.jpg" alt="Safe Image">
`
        })
      })

      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )

      // Dangerous elements should be removed
      await expect(helpPage.locator('script')).toHaveCount(0)
      await expect(helpPage.locator('iframe')).toHaveCount(0)

      // Check that onerror attribute is removed
      const images = helpPage.locator('img')
      const imageCount = await images.count()
      for (let i = 0; i < imageCount; i++) {
        const img = images.nth(i)
        const onError = await img.getAttribute('onerror')
        expect(onError).toBeNull()
      }

      // Check that javascript: links are sanitized
      const links = helpPage.locator('a')
      const linkCount = await links.count()
      for (let i = 0; i < linkCount; i++) {
        const link = links.nth(i)
        const href = await link.getAttribute('href')
        if (href !== null) {
          expect(href).not.toContain('javascript:')
        }
      }

      // Safe content should remain
      await expect(helpPage.locator('video[src*="safe.mp4"]')).toBeVisible()
      await expect(helpPage.locator('img[alt="Safe Image"]')).toBeVisible()
    })

    test('Should handle locale-specific documentation', async ({
      comfyPage
    }) => {
      // Mock different responses for different locales
      await comfyPage.page.route('**/docs/KSampler/ja.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: `# KSamplerノード

これは日本語のドキュメントです。
`
        })
      })

      await comfyPage.page.route('**/docs/KSampler/en.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: `# KSampler Node

This is English documentation.
`
        })
      })

      // Set locale to Japanese
      await comfyPage.settings.setSetting('Comfy.Locale', 'ja')

      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )
      await expect(helpPage).toContainText('KSamplerノード')
      await expect(helpPage).toContainText('これは日本語のドキュメントです')

      // Reset locale
      await comfyPage.settings.setSetting('Comfy.Locale', 'en')
    })

    test('Should handle network errors gracefully', async ({ comfyPage }) => {
      // Mock network error
      await comfyPage.page.route('**/docs/**/*.md', async (route) => {
        await route.abort('failed')
      })

      await comfyPage.workflow.loadWorkflow('default')
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )

      // Should show fallback content (node description)
      await expect(helpPage).toBeVisible()
      await expect(helpPage.locator('.p-progressspinner')).not.toBeVisible()

      // Should show some content even on error
      const content = await helpPage.textContent()
      expect(content).toBeTruthy()
    })

    test('Should update help content when switching between nodes', async ({
      comfyPage
    }) => {
      // Mock different help content for different nodes
      await comfyPage.page.route('**/docs/KSampler/en.md', async (route) => {
        await route.fulfill({
          status: 200,
          body: '# KSampler Help\n\nThis is KSampler documentation.'
        })
      })

      await comfyPage.page.route(
        '**/docs/CheckpointLoaderSimple/en.md',
        async (route) => {
          await route.fulfill({
            status: 200,
            body: '# Checkpoint Loader Help\n\nThis is Checkpoint Loader documentation.'
          })
        }
      )

      await comfyPage.workflow.loadWorkflow('default')
      await fitToViewInstant(comfyPage)

      // Select KSampler first
      const ksamplerNodes =
        await comfyPage.nodeOps.getNodeRefsByType('KSampler')
      await selectNodeWithPan(comfyPage, ksamplerNodes[0])

      const helpButton = comfyPage.page.locator(
        '.selection-toolbox button[data-testid="info-button"]'
      )
      await helpButton.click()

      const helpPage = comfyPage.page.locator(
        '[data-testid="properties-panel"]'
      )
      await expect(helpPage).toContainText('KSampler Help')
      await expect(helpPage).toContainText('This is KSampler documentation')

      // Now select Checkpoint Loader
      const checkpointNodes = await comfyPage.nodeOps.getNodeRefsByType(
        'CheckpointLoaderSimple'
      )
      await selectNodeWithPan(comfyPage, checkpointNodes[0])

      // Content should update
      await expect(helpPage).toContainText('Checkpoint Loader Help')
      await expect(helpPage).toContainText(
        'This is Checkpoint Loader documentation'
      )
      await expect(helpPage).not.toContainText('KSampler documentation')
    })
  })
})
