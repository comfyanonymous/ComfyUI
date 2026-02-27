import { expect } from '@playwright/test'

import type { Settings } from '../../src/schemas/apiSchema'
import type { SettingParams } from '../../src/platform/settings/types'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'

/**
 * Type helper for test settings with arbitrary IDs.
 * Extensions can register settings with any ID, but SettingParams.id
 * is typed as keyof Settings for autocomplete. This helper allows
 * arbitrary IDs in tests while keeping type safety for other fields.
 */
type TestSettingId = keyof Settings

test.describe('Topbar commands', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
  })

  test('Should allow registering topbar commands', async ({ comfyPage }) => {
    await comfyPage.page.evaluate(() => {
      window.app!.registerExtension({
        name: 'TestExtension1',
        commands: [
          {
            id: 'foo',
            label: 'foo-command',
            function: () => {
              window.foo = true
            }
          }
        ],
        menuCommands: [
          {
            path: ['ext'],
            commands: ['foo']
          }
        ]
      })
    })

    await comfyPage.menu.topbar.triggerTopbarCommand(['ext', 'foo-command'])
    expect(await comfyPage.page.evaluate(() => window.foo)).toBe(true)
  })

  test('Should not allow register command defined in other extension', async ({
    comfyPage
  }) => {
    await comfyPage.command.registerCommand('foo', () => alert(1))
    await comfyPage.page.evaluate(() => {
      window.app!.registerExtension({
        name: 'TestExtension1',
        menuCommands: [
          {
            path: ['ext'],
            commands: ['foo']
          }
        ]
      })
    })

    const menuItem = comfyPage.menu.topbar.getMenuItem('ext')
    expect(await menuItem.count()).toBe(0)
  })

  test('Should allow registering keybindings', async ({ comfyPage }) => {
    await comfyPage.page.evaluate(() => {
      const app = window.app!
      app.registerExtension({
        name: 'TestExtension1',
        commands: [
          {
            id: 'TestCommand',
            function: () => {
              window.TestCommand = true
            }
          }
        ],
        keybindings: [
          {
            combo: { key: 'k' },
            commandId: 'TestCommand'
          }
        ]
      })
    })

    await comfyPage.page.keyboard.press('k')
    expect(await comfyPage.page.evaluate(() => window.TestCommand)).toBe(true)
  })

  test.describe('Settings', () => {
    test('Should allow adding settings', async ({ comfyPage }) => {
      await comfyPage.page.evaluate(() => {
        window.app!.registerExtension({
          name: 'TestExtension1',
          settings: [
            {
              // Extensions can register arbitrary setting IDs
              id: 'TestSetting' as TestSettingId,
              name: 'Test Setting',
              type: 'text',
              defaultValue: 'Hello, world!',
              onChange: () => {
                window.changeCount = (window.changeCount ?? 0) + 1
              }
            }
          ]
        })
      })
      // onChange is called when the setting is first added
      expect(await comfyPage.page.evaluate(() => window.changeCount)).toBe(1)
      expect(await comfyPage.settings.getSetting('TestSetting')).toBe(
        'Hello, world!'
      )

      await comfyPage.settings.setSetting('TestSetting', 'Hello, universe!')
      expect(await comfyPage.settings.getSetting('TestSetting')).toBe(
        'Hello, universe!'
      )
      expect(await comfyPage.page.evaluate(() => window.changeCount)).toBe(2)
    })

    test('Should allow setting boolean settings', async ({ comfyPage }) => {
      await comfyPage.page.evaluate(() => {
        window.app!.registerExtension({
          name: 'TestExtension1',
          settings: [
            {
              // Extensions can register arbitrary setting IDs
              id: 'Comfy.TestSetting' as TestSettingId,
              name: 'Test Setting',
              type: 'boolean',
              defaultValue: false,
              onChange: () => {
                window.changeCount = (window.changeCount ?? 0) + 1
              }
            }
          ]
        })
      })

      expect(await comfyPage.settings.getSetting('Comfy.TestSetting')).toBe(
        false
      )
      expect(await comfyPage.page.evaluate(() => window.changeCount)).toBe(1)

      await comfyPage.settingDialog.open()
      await comfyPage.settingDialog.toggleBooleanSetting('Comfy.TestSetting')
      expect(await comfyPage.settings.getSetting('Comfy.TestSetting')).toBe(
        true
      )
      expect(await comfyPage.page.evaluate(() => window.changeCount)).toBe(2)
    })

    test.describe('Passing through attrs to setting components', () => {
      const testCases: Array<{
        config: Pick<SettingParams, 'type' | 'defaultValue'> &
          Partial<Omit<SettingParams, 'id' | 'type' | 'defaultValue'>>
        selector: string
      }> = [
        {
          config: {
            type: 'boolean',
            defaultValue: true
          },
          selector: '.p-toggleswitch.p-component'
        },
        {
          config: {
            type: 'number',
            defaultValue: 10
          },
          selector: '.p-inputnumber input'
        },
        {
          config: {
            type: 'slider',
            defaultValue: 10
          },
          selector: '.p-slider.p-component'
        },
        {
          config: {
            type: 'combo',
            defaultValue: 'foo',
            options: ['foo', 'bar', 'baz']
          },
          selector: '.p-select.p-component'
        },
        {
          config: {
            type: 'text',
            defaultValue: 'Hello'
          },
          selector: '.p-inputtext'
        },
        {
          config: {
            type: 'color',
            defaultValue: '#000000'
          },
          selector: '.p-colorpicker-preview'
        }
      ] as const

      for (const { config, selector } of testCases) {
        test(`${config.type} component should respect disabled attr`, async ({
          comfyPage
        }) => {
          await comfyPage.page.evaluate((config) => {
            window.app!.registerExtension({
              name: 'TestExtension1',
              settings: [
                {
                  // Extensions can register arbitrary setting IDs
                  id: 'Comfy.TestSetting' as TestSettingId,
                  name: 'Test',
                  attrs: { disabled: true },
                  ...config
                }
              ]
            })
          }, config)

          await comfyPage.settingDialog.open()
          const component = comfyPage.settingDialog.root
            .getByText('TestSetting Test')
            .locator(selector)

          const isDisabled = await component.evaluate((el) =>
            el.tagName === 'INPUT'
              ? (el as HTMLInputElement).disabled
              : el.classList.contains('p-disabled')
          )
          expect(isDisabled).toBe(true)
        })
      }
    })
  })

  test.describe('About panel', () => {
    test('Should allow adding badges', async ({ comfyPage }) => {
      await comfyPage.page.evaluate(() => {
        window.app!.registerExtension({
          name: 'TestExtension1',
          aboutPageBadges: [
            {
              label: 'Test Badge',
              url: 'https://example.com',
              icon: 'pi pi-box'
            }
          ]
        })
      })

      await comfyPage.settingDialog.open()
      await comfyPage.settingDialog.goToAboutPanel()
      const badge = comfyPage.page.locator('.about-badge').last()
      expect(badge).toBeDefined()
      expect(await badge.textContent()).toContain('Test Badge')
    })
  })

  test.describe('Dialog', () => {
    test('Should allow showing a prompt dialog', async ({ comfyPage }) => {
      await comfyPage.page.evaluate(() => {
        void window
          .app!.extensionManager.dialog.prompt({
            title: 'Test Prompt',
            message: 'Test Prompt Message'
          })
          .then((value: string | null) => {
            ;(window as unknown as Record<string, unknown>)['value'] = value
          })
      })

      await comfyPage.nodeOps.fillPromptDialog('Hello, world!')
      expect(
        await comfyPage.page.evaluate(
          () => (window as unknown as Record<string, unknown>)['value']
        )
      ).toBe('Hello, world!')
    })

    test('Should allow showing a confirmation dialog', async ({
      comfyPage
    }) => {
      await comfyPage.page.evaluate(() => {
        void window
          .app!.extensionManager.dialog.confirm({
            title: 'Test Confirm',
            message: 'Test Confirm Message'
          })
          .then((value: boolean | null) => {
            ;(window as unknown as Record<string, unknown>)['value'] = value
          })
      })

      await comfyPage.confirmDialog.click('confirm')
      expect(
        await comfyPage.page.evaluate(
          () => (window as unknown as Record<string, unknown>)['value']
        )
      ).toBe(true)
    })

    test('Should allow dismissing a dialog', async ({ comfyPage }) => {
      await comfyPage.page.evaluate(() => {
        ;(window as unknown as Record<string, unknown>)['value'] = 'foo'
        void window
          .app!.extensionManager.dialog.confirm({
            title: 'Test Confirm',
            message: 'Test Confirm Message'
          })
          .then((value: boolean | null) => {
            ;(window as unknown as Record<string, unknown>)['value'] = value
          })
      })

      await comfyPage.confirmDialog.click('reject')
      expect(
        await comfyPage.page.evaluate(
          () => (window as unknown as Record<string, unknown>)['value']
        )
      ).toBeNull()
    })
  })

  test.describe('Selection Toolbox', () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.Canvas.SelectionToolbox', true)
    })

    test('Should allow adding commands to selection toolbox', async ({
      comfyPage
    }) => {
      // Register an extension with a selection toolbox command
      await comfyPage.page.evaluate(() => {
        window.app!.registerExtension({
          name: 'TestExtension1',
          commands: [
            {
              id: 'test.selection.command',
              label: 'Test Command',
              icon: 'pi pi-star',
              function: () => {
                ;(window as unknown as Record<string, unknown>)[
                  'selectionCommandExecuted'
                ] = true
              }
            }
          ],
          getSelectionToolboxCommands: () => ['test.selection.command']
        })
      })

      await comfyPage.nodeOps.selectNodes(['CLIP Text Encode (Prompt)'])

      // Click the command button in the selection toolbox
      const toolboxButton = comfyPage.page.locator(
        '.selection-toolbox button:has(.pi-star)'
      )
      await toolboxButton.click()

      expect(
        await comfyPage.page.evaluate(
          () =>
            (window as unknown as Record<string, unknown>)[
              'selectionCommandExecuted'
            ]
        )
      ).toBe(true)
    })
  })
})
