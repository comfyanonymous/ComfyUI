import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import type { WorkspaceStore } from '../types/globals'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

const customColorPalettes = {
  obsidian: {
    version: 102,
    id: 'obsidian',
    name: 'Obsidian',
    colors: {
      node_slot: {
        CLIP: '#FFD500',
        CLIP_VISION: '#A8DADC',
        CLIP_VISION_OUTPUT: '#ad7452',
        CONDITIONING: '#FFA931',
        CONTROL_NET: '#6EE7B7',
        IMAGE: '#64B5F6',
        LATENT: '#FF9CF9',
        MASK: '#81C784',
        MODEL: '#B39DDB',
        STYLE_MODEL: '#C2FFAE',
        VAE: '#FF6E6E',
        TAESD: '#DCC274',
        PIPE_LINE: '#7737AA',
        PIPE_LINE_SDXL: '#7737AA',
        INT: '#29699C',
        XYPLOT: '#74DA5D',
        X_Y: '#38291f'
      },
      litegraph_base: {
        BACKGROUND_IMAGE:
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAQBJREFUeNrs1rEKwjAUhlETUkj3vP9rdmr1Ysammk2w5wdxuLgcMHyptfawuZX4pJSWZTnfnu/lnIe/jNNxHHGNn//HNbbv+4dr6V+11uF527arU7+u63qfa/bnmh8sWLBgwYJlqRf8MEptXPBXJXa37BSl3ixYsGDBMliwFLyCV/DeLIMFCxYsWLBMwSt4Be/NggXLYMGCBUvBK3iNruC9WbBgwYJlsGApeAWv4L1ZBgsWLFiwYJmCV/AK3psFC5bBggULloJX8BpdwXuzYMGCBctgwVLwCl7Be7MMFixYsGDBsu8FH1FaSmExVfAxBa/gvVmwYMGCZbBg/W4vAQYA5tRF9QYlv/QAAAAASUVORK5CYII=',
        CLEAR_BACKGROUND_COLOR: '#222222',
        NODE_TITLE_COLOR: 'rgba(255,255,255,.75)',
        NODE_SELECTED_TITLE_COLOR: '#FFF',
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: '#b8b8b8',
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: 'rgba(0,0,0,.8)',
        NODE_DEFAULT_BGCOLOR: 'rgba(22,22,22,.8)',
        NODE_DEFAULT_BOXCOLOR: 'rgba(255,255,255,.75)',
        NODE_DEFAULT_SHAPE: 'box',
        NODE_BOX_OUTLINE_COLOR: '#236692',
        DEFAULT_SHADOW_COLOR: 'rgba(0,0,0,0)',
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: '#242424',
        WIDGET_OUTLINE_COLOR: '#333',
        WIDGET_TEXT_COLOR: '#a3a3a8',
        WIDGET_SECONDARY_TEXT_COLOR: '#97979c',
        WIDGET_DISABLED_TEXT_COLOR: '#646464',
        LINK_COLOR: '#9A9',
        EVENT_LINK_COLOR: '#A86',
        CONNECTING_LINK_COLOR: '#AFA'
      },
      comfy_base: {
        'fg-color': '#fff',
        'bg-color': '#242424',
        'comfy-menu-bg': 'rgba(24,24,24,.9)',
        'comfy-input-bg': '#262626',
        'input-text': '#ddd',
        'descrip-text': '#999',
        'drag-text': '#ccc',
        'error-text': '#ff4444',
        'border-color': '#29292c',
        'tr-even-bg-color': 'rgba(28,28,28,.9)',
        'tr-odd-bg-color': 'rgba(19,19,19,.9)'
      }
    }
  },
  obsidian_dark: {
    version: 102,
    id: 'obsidian_dark',
    name: 'Obsidian Dark',
    colors: {
      node_slot: {
        CLIP: '#FFD500',
        CLIP_VISION: '#A8DADC',
        CLIP_VISION_OUTPUT: '#ad7452',
        CONDITIONING: '#FFA931',
        CONTROL_NET: '#6EE7B7',
        IMAGE: '#64B5F6',
        LATENT: '#FF9CF9',
        MASK: '#81C784',
        MODEL: '#B39DDB',
        STYLE_MODEL: '#C2FFAE',
        VAE: '#FF6E6E',
        TAESD: '#DCC274',
        PIPE_LINE: '#7737AA',
        PIPE_LINE_SDXL: '#7737AA',
        INT: '#29699C',
        XYPLOT: '#74DA5D',
        X_Y: '#38291f'
      },
      litegraph_base: {
        BACKGROUND_IMAGE:
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAACXBIWXMAAAsTAAALEwEAmpwYAAAGlmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgOS4xLWMwMDEgNzkuMTQ2Mjg5OSwgMjAyMy8wNi8yNS0yMDowMTo1NSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMy0xMS0xM1QwMDoxODowMiswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOmIyYzRhNjA5LWJmYTctYTg0MC1iOGFlLTk3MzE2ZjM1ZGIyNyIgeG1wTU06RG9jdW1lbnRJRD0iYWRvYmU6ZG9jaWQ6cGhvdG9zaG9wOjk0ZmNlZGU4LTE1MTctZmQ0MC04ZGU3LWYzOTgxM2E3ODk5ZiIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjIzMWIxMGIwLWI0ZmItMDI0ZS1iMTJlLTMwNTMwM2NkMDdjOCI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249ImNyZWF0ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6MjMxYjEwYjAtYjRmYi0wMjRlLWIxMmUtMzA1MzAzY2QwN2M4IiBzdEV2dDp3aGVuPSIyMDIzLTExLTEzVDAwOjE4OjAyKzAxOjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjUuMSAoV2luZG93cykiLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjQ4OWY1NzlmLTJkNjUtZWQ0Zi04OTg0LTA4NGE2MGE1ZTMzNSIgc3RFdnQ6d2hlbj0iMjAyMy0xMS0xNVQwMjowNDo1OSswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDI1LjEgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDpiMmM0YTYwOS1iZmE3LWE4NDAtYjhhZS05NzMxNmYzNWRiMjciIHN0RXZ0OndoZW49IjIwMjMtMTEtMTVUMDI6MDQ6NTkrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyNS4xIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz4OTe6GAAAAx0lEQVR42u3WMQoAIQxFwRzJys77X8vSLiRgITif7bYbgrwYc/mKXyBoY4VVBgsWLFiwYFmOlTv+9jfDOjHmr8u6eVkGCxYsWLBgmc5S8ApewXvgYRksWLBgKXidpeBdloL3wMOCBctgwVLwCl7BuyyDBQsWLFiwTGcpeAWv4D3wsAwWLFiwFLzOUvAuS8F74GHBgmWwYCl4Ba/gXZbBggULFixYprMUvIJX8B54WAYLFixYCl5nKXiXpeA98LBgwTJYsGC9tg1o8f4TTtqzNQAAAABJRU5ErkJggg==',
        CLEAR_BACKGROUND_COLOR: '#000',
        NODE_TITLE_COLOR: 'rgba(255,255,255,.75)',
        NODE_SELECTED_TITLE_COLOR: '#FFF',
        NODE_TEXT_SIZE: 14,
        NODE_TEXT_COLOR: '#b8b8b8',
        NODE_SUBTEXT_SIZE: 12,
        NODE_DEFAULT_COLOR: 'rgba(0,0,0,.8)',
        NODE_DEFAULT_BGCOLOR: 'rgba(22,22,22,.8)',
        NODE_DEFAULT_BOXCOLOR: 'rgba(255,255,255,.75)',
        NODE_DEFAULT_SHAPE: 'box',
        NODE_BOX_OUTLINE_COLOR: '#236692',
        DEFAULT_SHADOW_COLOR: 'rgba(0,0,0,0)',
        DEFAULT_GROUP_FONT: 24,
        WIDGET_BGCOLOR: '#242424',
        WIDGET_OUTLINE_COLOR: '#333',
        WIDGET_TEXT_COLOR: '#a3a3a8',
        WIDGET_SECONDARY_TEXT_COLOR: '#97979c',
        WIDGET_DISABLED_TEXT_COLOR: '#646464',
        LINK_COLOR: '#9A9',
        EVENT_LINK_COLOR: '#A86',
        CONNECTING_LINK_COLOR: '#AFA'
      },
      comfy_base: {
        'fg-color': '#fff',
        'bg-color': '#242424',
        'comfy-menu-bg': 'rgba(24,24,24,.9)',
        'comfy-input-bg': '#262626',
        'input-text': '#ddd',
        'descrip-text': '#999',
        'drag-text': '#ccc',
        'error-text': '#ff4444',
        'border-color': '#29292c',
        'tr-even-bg-color': 'rgba(28,28,28,.9)',
        'tr-odd-bg-color': 'rgba(19,19,19,.9)'
      }
    }
  },
  // A custom light theme with fg color red
  light_red: {
    id: 'light_red',
    name: 'Light Red',
    light_theme: true,
    colors: {
      node_slot: {},
      litegraph_base: {},
      comfy_base: {
        'fg-color': '#ff0000'
      }
    }
  }
}

test.describe('Color Palette', { tag: ['@screenshot', '@settings'] }, () => {
  test('Can show custom color palette', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting(
      'Comfy.CustomColorPalettes',
      customColorPalettes
    )
    // Reload to apply the new setting. Setting Comfy.CustomColorPalettes directly
    // doesn't update the store immediately.
    await comfyPage.setup()

    await comfyPage.workflow.loadWorkflow('nodes/every_node_color')
    await comfyPage.settings.setSetting('Comfy.ColorPalette', 'obsidian_dark')
    await expect(comfyPage.canvas).toHaveScreenshot(
      'custom-color-palette-obsidian-dark-all-colors.png'
    )
    await comfyPage.settings.setSetting('Comfy.ColorPalette', 'light_red')
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'custom-color-palette-light-red.png'
    )

    await comfyPage.settings.setSetting('Comfy.ColorPalette', 'dark')
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot('default-color-palette.png')
  })

  test('Can add custom color palette', async ({ comfyPage }) => {
    await comfyPage.page.evaluate(async (p) => {
      await (
        window.app!.extensionManager as WorkspaceStore
      ).colorPalette.addCustomColorPalette(p)
    }, customColorPalettes.obsidian_dark)
    expect(await comfyPage.toast.getToastErrorCount()).toBe(0)

    await comfyPage.settings.setSetting('Comfy.ColorPalette', 'obsidian_dark')
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'custom-color-palette-obsidian-dark.png'
    )
    // Legacy `custom_` prefix is still supported
    await comfyPage.settings.setSetting(
      'Comfy.ColorPalette',
      'custom_obsidian_dark'
    )
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'custom-color-palette-obsidian-dark.png'
    )
  })
})

test.describe(
  'Node Color Adjustments',
  { tag: ['@screenshot', '@settings'] },
  () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.workflow.loadWorkflow('nodes/every_node_color')
    })

    test('should adjust opacity via node opacity setting', async ({
      comfyPage
    }) => {
      await comfyPage.settings.setSetting('Comfy.Node.Opacity', 0.5)

      // Drag mouse to force canvas to redraw
      await comfyPage.page.mouse.move(0, 0)

      await expect(comfyPage.canvas).toHaveScreenshot('node-opacity-0.5.png')

      await comfyPage.settings.setSetting('Comfy.Node.Opacity', 1.0)

      await comfyPage.page.mouse.move(8, 8)
      await expect(comfyPage.canvas).toHaveScreenshot('node-opacity-1.png')
    })

    test('should persist color adjustments when changing themes', async ({
      comfyPage
    }) => {
      await comfyPage.settings.setSetting('Comfy.Node.Opacity', 0.2)
      await comfyPage.settings.setSetting('Comfy.ColorPalette', 'arc')
      await comfyPage.nextFrame()
      await comfyPage.page.mouse.move(0, 0)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'node-opacity-0.2-arc-theme.png'
      )
    })

    test('should not serialize color adjustments in workflow', async ({
      comfyPage
    }) => {
      await comfyPage.settings.setSetting('Comfy.Node.Opacity', 0.5)
      await comfyPage.settings.setSetting('Comfy.ColorPalette', 'light')
      await comfyPage.nextFrame()
      const parsed = await comfyPage.page.evaluate(() => {
        const graph = window.app!.graph!
        if (typeof graph.serialize !== 'function') {
          throw new Error('app.graph.serialize is not available')
        }
        return graph.serialize() as {
          nodes: Array<{ bgcolor?: string; color?: string }>
        }
      })
      expect(parsed.nodes).toBeDefined()
      expect(Array.isArray(parsed.nodes)).toBe(true)
      const nodes = parsed.nodes
      for (const node of nodes) {
        if (node.bgcolor) expect(node.bgcolor).not.toMatch(/hsla/)
        if (node.color) expect(node.color).not.toMatch(/hsla/)
      }
    })

    test('should lighten node colors when switching to light theme', async ({
      comfyPage
    }) => {
      await comfyPage.settings.setSetting('Comfy.ColorPalette', 'light')
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot(
        'node-lightened-colors.png'
      )
    })

    test.describe('Context menu color adjustments', () => {
      test.beforeEach(async ({ comfyPage }) => {
        await comfyPage.settings.setSetting('Comfy.ColorPalette', 'light')
        await comfyPage.settings.setSetting('Comfy.Node.Opacity', 0.3)
        const node = await comfyPage.nodeOps.getFirstNodeRef()
        await node?.clickContextMenuOption('Colors')
      })

      test('should persist color adjustments when changing custom node colors', async ({
        comfyPage
      }) => {
        await comfyPage.page
          .locator('.litemenu-entry.submenu span:has-text("red")')
          .click()
        await expect(comfyPage.canvas).toHaveScreenshot(
          'node-opacity-0.3-color-changed.png'
        )
      })

      test('should persist color adjustments when removing custom node color', async ({
        comfyPage
      }) => {
        await comfyPage.page
          .locator('.litemenu-entry.submenu span:has-text("No color")')
          .click()
        await expect(comfyPage.canvas).toHaveScreenshot(
          'node-opacity-0.3-color-removed.png'
        )
      })
    })
  }
)
