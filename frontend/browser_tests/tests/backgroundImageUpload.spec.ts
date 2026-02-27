import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Background Image Upload', () => {
  test.beforeEach(async ({ comfyPage }) => {
    // Reset the background image setting before each test
    await comfyPage.settings.setSetting('Comfy.Canvas.BackgroundImage', '')
  })

  test.afterEach(async ({ comfyPage }) => {
    // Clean up background image setting after each test
    await comfyPage.settings.setSetting('Comfy.Canvas.BackgroundImage', '')
  })

  test('should show background image upload component in settings', async ({
    comfyPage
  }) => {
    // Open settings dialog
    await comfyPage.page.keyboard.press('Control+,')

    // Navigate to Appearance category
    const appearanceOption = comfyPage.page.locator('text=Appearance')
    await appearanceOption.click()

    // Find the background image setting
    const backgroundImageSetting = comfyPage.page.locator(
      '#Comfy\\.Canvas\\.BackgroundImage'
    )
    await expect(backgroundImageSetting).toBeVisible()

    // Verify the component has the expected elements using semantic selectors
    const urlInput = backgroundImageSetting.getByRole('textbox')
    await expect(urlInput).toBeVisible()
    await expect(urlInput).toHaveAttribute('placeholder')

    const uploadButton = backgroundImageSetting.getByRole('button', {
      name: /upload/i
    })
    await expect(uploadButton).toBeVisible()

    const clearButton = backgroundImageSetting.getByRole('button', {
      name: /clear/i
    })
    await expect(clearButton).toBeVisible()
    await expect(clearButton).toBeDisabled() // Should be disabled when no image
  })

  test('should upload image file and set as background', async ({
    comfyPage
  }) => {
    // Open settings dialog
    await comfyPage.page.keyboard.press('Control+,')

    // Navigate to Appearance category
    const appearanceOption = comfyPage.page.locator('text=Appearance')
    await appearanceOption.click()

    // Find the background image setting
    const backgroundImageSetting = comfyPage.page.locator(
      '#Comfy\\.Canvas\\.BackgroundImage'
    )
    // Click the upload button to trigger file input
    const uploadButton = backgroundImageSetting.getByRole('button', {
      name: /upload/i
    })

    // Set up file upload handler
    const fileChooserPromise = comfyPage.page.waitForEvent('filechooser')
    await uploadButton.click()
    const fileChooser = await fileChooserPromise

    // Upload the test image
    await fileChooser.setFiles(comfyPage.assetPath('image32x32.webp'))

    // Verify the URL input now has an API URL
    const urlInput = backgroundImageSetting.getByRole('textbox')
    await expect(urlInput).toHaveValue(/^\/api\/view\?.*subfolder=backgrounds/)

    // Verify clear button is now enabled
    const clearButton = backgroundImageSetting.getByRole('button', {
      name: /clear/i
    })
    await expect(clearButton).toBeEnabled()

    // Verify the setting value was actually set
    const settingValue = await comfyPage.settings.getSetting(
      'Comfy.Canvas.BackgroundImage'
    )
    expect(settingValue).toMatch(/^\/api\/view\?.*subfolder=backgrounds/)
  })

  test('should accept URL input for background image', async ({
    comfyPage
  }) => {
    const testImageUrl = 'https://example.com/test-image.png'

    // Open settings dialog
    await comfyPage.page.keyboard.press('Control+,')

    // Navigate to Appearance category
    const appearanceOption = comfyPage.page.locator('text=Appearance')
    await appearanceOption.click()

    // Find the background image setting
    const backgroundImageSetting = comfyPage.page.locator(
      '#Comfy\\.Canvas\\.BackgroundImage'
    )
    // Enter URL in the input field
    const urlInput = backgroundImageSetting.getByRole('textbox')
    await urlInput.fill(testImageUrl)

    // Trigger blur event to ensure the value is set
    await urlInput.blur()

    // Verify clear button is now enabled
    const clearButton = backgroundImageSetting.getByRole('button', {
      name: /clear/i
    })
    await expect(clearButton).toBeEnabled()

    // Verify the setting value was updated
    const settingValue = await comfyPage.settings.getSetting(
      'Comfy.Canvas.BackgroundImage'
    )
    expect(settingValue).toBe(testImageUrl)
  })

  test('should clear background image when clear button is clicked', async ({
    comfyPage
  }) => {
    const testImageUrl = 'https://example.com/test-image.png'

    // First set a background image
    await comfyPage.settings.setSetting(
      'Comfy.Canvas.BackgroundImage',
      testImageUrl
    )

    // Open settings dialog
    await comfyPage.page.keyboard.press('Control+,')

    // Navigate to Appearance category
    const appearanceOption = comfyPage.page.locator('text=Appearance')
    await appearanceOption.click()

    // Find the background image setting
    const backgroundImageSetting = comfyPage.page.locator(
      '#Comfy\\.Canvas\\.BackgroundImage'
    )
    // Verify the input has the test URL
    const urlInput = backgroundImageSetting.getByRole('textbox')
    await expect(urlInput).toHaveValue(testImageUrl)

    // Verify clear button is enabled
    const clearButton = backgroundImageSetting.getByRole('button', {
      name: /clear/i
    })
    await expect(clearButton).toBeEnabled()

    // Click the clear button
    await clearButton.click()

    // Verify the input is now empty
    await expect(urlInput).toHaveValue('')

    // Verify clear button is now disabled
    await expect(clearButton).toBeDisabled()

    // Verify the setting value was cleared
    const settingValue = await comfyPage.settings.getSetting(
      'Comfy.Canvas.BackgroundImage'
    )
    expect(settingValue).toBe('')
  })

  test('should show tooltip on upload and clear buttons', async ({
    comfyPage
  }) => {
    // Open settings dialog
    await comfyPage.page.keyboard.press('Control+,')

    // Navigate to Appearance category
    const appearanceOption = comfyPage.page.locator('text=Appearance')
    await appearanceOption.click()

    // Find the background image setting
    const backgroundImageSetting = comfyPage.page.locator(
      '#Comfy\\.Canvas\\.BackgroundImage'
    )
    // Hover over upload button and verify tooltip appears
    const uploadButton = backgroundImageSetting.getByRole('button', {
      name: /upload/i
    })
    await uploadButton.hover()

    const uploadTooltip = comfyPage.page.locator('.p-tooltip:visible')
    await expect(uploadTooltip).toBeVisible()

    // Move away to hide tooltip
    await comfyPage.page.locator('body').hover()

    // Set a background to enable clear button
    const urlInput = backgroundImageSetting.getByRole('textbox')
    await urlInput.fill('https://example.com/test.png')
    await urlInput.blur()

    // Hover over clear button and verify tooltip appears
    const clearButton = backgroundImageSetting.getByRole('button', {
      name: /clear/i
    })
    await clearButton.hover()

    const clearTooltip = comfyPage.page.locator('.p-tooltip:visible')
    await expect(clearTooltip).toBeVisible()
  })

  test('should maintain reactive updates between URL input and clear button state', async ({
    comfyPage
  }) => {
    // Open settings dialog
    await comfyPage.page.keyboard.press('Control+,')

    // Navigate to Appearance category
    const appearanceOption = comfyPage.page.locator('text=Appearance')
    await appearanceOption.click()

    // Find the background image setting
    const backgroundImageSetting = comfyPage.page.locator(
      '#Comfy\\.Canvas\\.BackgroundImage'
    )
    const urlInput = backgroundImageSetting.getByRole('textbox')
    const clearButton = backgroundImageSetting.getByRole('button', {
      name: /clear/i
    })

    // Initially clear button should be disabled
    await expect(clearButton).toBeDisabled()

    // Type some text - clear button should become enabled
    await urlInput.fill('test')
    await expect(clearButton).toBeEnabled()

    // Clear the text manually - clear button should become disabled again
    await urlInput.fill('')
    await expect(clearButton).toBeDisabled()

    // Add text again - clear button should become enabled
    await urlInput.fill('https://example.com/image.png')
    await expect(clearButton).toBeEnabled()

    // Use clear button - should clear input and disable itself
    await clearButton.click()
    await expect(urlInput).toHaveValue('')
    await expect(clearButton).toBeDisabled()
  })
})
