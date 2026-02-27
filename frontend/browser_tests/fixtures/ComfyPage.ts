import type {
  APIRequestContext,
  ExpectMatcherState,
  Locator,
  Page
} from '@playwright/test'
import { test as base, expect } from '@playwright/test'
import dotenv from 'dotenv'

import { TestIds } from './selectors'
import { NodeBadgeMode } from '../../src/types/nodeSource'
import { ComfyActionbar } from '../helpers/actionbar'
import { ComfyTemplates } from '../helpers/templates'
import { ComfyMouse } from './ComfyMouse'
import { VueNodeHelpers } from './VueNodeHelpers'
import { ComfyNodeSearchBox } from './components/ComfyNodeSearchBox'
import { ComfyNodeSearchBoxV2 } from './components/ComfyNodeSearchBoxV2'
import { ContextMenu } from './components/ContextMenu'
import { SettingDialog } from './components/SettingDialog'
import { BottomPanel } from './components/BottomPanel'
import {
  NodeLibrarySidebarTab,
  WorkflowsSidebarTab
} from './components/SidebarTab'
import { Topbar } from './components/Topbar'
import { CanvasHelper } from './helpers/CanvasHelper'
import { PerformanceHelper } from './helpers/PerformanceHelper'
import { ClipboardHelper } from './helpers/ClipboardHelper'
import { CommandHelper } from './helpers/CommandHelper'
import { DragDropHelper } from './helpers/DragDropHelper'
import { KeyboardHelper } from './helpers/KeyboardHelper'
import { NodeOperationsHelper } from './helpers/NodeOperationsHelper'
import { SettingsHelper } from './helpers/SettingsHelper'
import { SubgraphHelper } from './helpers/SubgraphHelper'
import { ToastHelper } from './helpers/ToastHelper'
import { WorkflowHelper } from './helpers/WorkflowHelper'
import type { NodeReference } from './utils/litegraphUtils'
import type { WorkspaceStore } from '../types/globals'

dotenv.config()

class ComfyPropertiesPanel {
  readonly root: Locator
  readonly panelTitle: Locator
  readonly searchBox: Locator

  constructor(readonly page: Page) {
    this.root = page.getByTestId(TestIds.propertiesPanel.root)
    this.panelTitle = this.root.locator('h3')
    this.searchBox = this.root.getByPlaceholder(/^Search/)
  }
}

class ComfyMenu {
  private _nodeLibraryTab: NodeLibrarySidebarTab | null = null
  private _workflowsTab: WorkflowsSidebarTab | null = null
  private _topbar: Topbar | null = null

  public readonly sideToolbar: Locator
  public readonly propertiesPanel: ComfyPropertiesPanel
  public readonly modeToggleButton: Locator

  constructor(public readonly page: Page) {
    this.sideToolbar = page.getByTestId(TestIds.sidebar.toolbar)
    this.modeToggleButton = page.getByTestId(TestIds.sidebar.modeToggle)
    this.propertiesPanel = new ComfyPropertiesPanel(page)
  }

  get buttons() {
    return this.sideToolbar.locator('.side-bar-button')
  }

  get nodeLibraryTab() {
    this._nodeLibraryTab ??= new NodeLibrarySidebarTab(this.page)
    return this._nodeLibraryTab
  }

  get workflowsTab() {
    this._workflowsTab ??= new WorkflowsSidebarTab(this.page)
    return this._workflowsTab
  }

  get topbar() {
    this._topbar ??= new Topbar(this.page)
    return this._topbar
  }

  async toggleTheme() {
    const currentTheme = await this.getThemeId()
    await this.modeToggleButton.click()
    await this.page.waitForFunction(
      (prevTheme) => {
        const settings = window.app?.ui?.settings
        return (
          settings &&
          settings.getSettingValue('Comfy.ColorPalette') !== prevTheme
        )
      },
      currentTheme,
      { timeout: 5000 }
    )
  }

  async getThemeId() {
    return await this.page.evaluate(async () => {
      return await window.app!.ui.settings.getSettingValue('Comfy.ColorPalette')
    })
  }
}

type KeysOfType<T, Match> = {
  [K in keyof T]: T[K] extends Match ? K : never
}[keyof T]

class ConfirmDialog {
  private readonly root: Locator
  public readonly delete: Locator
  public readonly overwrite: Locator
  public readonly reject: Locator
  public readonly confirm: Locator

  constructor(public readonly page: Page) {
    this.root = page.getByRole('dialog')
    this.delete = this.root.getByRole('button', { name: 'Delete' })
    this.overwrite = this.root.getByRole('button', { name: 'Overwrite' })
    this.reject = this.root.getByRole('button', { name: 'Cancel' })
    this.confirm = this.root.getByRole('button', { name: 'Confirm' })
  }

  async click(locator: KeysOfType<ConfirmDialog, Locator>) {
    const loc = this[locator]
    await loc.waitFor({ state: 'visible' })
    await loc.click()

    // Wait for the dialog mask to disappear after confirming
    const mask = this.page.locator('.p-dialog-mask')
    const count = await mask.count()
    if (count > 0) {
      await mask.first().waitFor({ state: 'hidden', timeout: 3000 })
    }

    // Wait for workflow service to finish if it's busy
    await this.page.waitForFunction(
      () =>
        (window.app?.extensionManager as WorkspaceStore | undefined)?.workflow
          ?.isBusy === false,
      undefined,
      { timeout: 3000 }
    )
  }
}

export class ComfyPage {
  public readonly url: string
  // All canvas position operations are based on default view of canvas.
  public readonly canvas: Locator
  public readonly selectionToolbox: Locator
  public readonly widgetTextBox: Locator

  // Buttons
  public readonly resetViewButton: Locator
  public readonly queueButton: Locator // Run button in Legacy UI
  public readonly runButton: Locator // Run button (renamed "Queue" -> "Run")

  // Inputs
  public readonly workflowUploadInput: Locator

  // Components
  public readonly searchBox: ComfyNodeSearchBox
  public readonly searchBoxV2: ComfyNodeSearchBoxV2
  public readonly menu: ComfyMenu
  public readonly actionbar: ComfyActionbar
  public readonly templates: ComfyTemplates
  public readonly settingDialog: SettingDialog
  public readonly confirmDialog: ConfirmDialog
  public readonly vueNodes: VueNodeHelpers
  public readonly subgraph: SubgraphHelper
  public readonly canvasOps: CanvasHelper
  public readonly nodeOps: NodeOperationsHelper
  public readonly settings: SettingsHelper
  public readonly keyboard: KeyboardHelper
  public readonly clipboard: ClipboardHelper
  public readonly workflow: WorkflowHelper
  public readonly contextMenu: ContextMenu
  public readonly toast: ToastHelper
  public readonly dragDrop: DragDropHelper
  public readonly command: CommandHelper
  public readonly bottomPanel: BottomPanel
  public readonly perf: PerformanceHelper

  /** Worker index to test user ID */
  public readonly userIds: string[] = []

  /** Test user ID for the current context */
  get id() {
    return this.userIds[comfyPageFixture.info().parallelIndex]
  }

  constructor(
    public readonly page: Page,
    public readonly request: APIRequestContext
  ) {
    this.url = process.env.PLAYWRIGHT_TEST_URL || 'http://localhost:8188'
    this.canvas = page.locator('#graph-canvas')
    this.selectionToolbox = page.locator('.selection-toolbox')
    this.widgetTextBox = page.getByPlaceholder('text').nth(1)
    this.resetViewButton = page.getByRole('button', { name: 'Reset View' })
    this.queueButton = page.getByRole('button', { name: 'Queue Prompt' })
    this.runButton = page
      .getByTestId(TestIds.topbar.queueButton)
      .getByRole('button', { name: 'Run' })
    this.workflowUploadInput = page.locator('#comfy-file-input')

    this.searchBox = new ComfyNodeSearchBox(page)
    this.searchBoxV2 = new ComfyNodeSearchBoxV2(page)
    this.menu = new ComfyMenu(page)
    this.actionbar = new ComfyActionbar(page)
    this.templates = new ComfyTemplates(page)
    this.settingDialog = new SettingDialog(page, this)
    this.confirmDialog = new ConfirmDialog(page)
    this.vueNodes = new VueNodeHelpers(page)
    this.subgraph = new SubgraphHelper(this)
    this.canvasOps = new CanvasHelper(page, this.canvas, this.resetViewButton)
    this.nodeOps = new NodeOperationsHelper(this)
    this.settings = new SettingsHelper(page)
    this.keyboard = new KeyboardHelper(page, this.canvas)
    this.clipboard = new ClipboardHelper(this.keyboard)
    this.workflow = new WorkflowHelper(this)
    this.contextMenu = new ContextMenu(page)
    this.toast = new ToastHelper(page)
    this.dragDrop = new DragDropHelper(page, this.assetPath.bind(this))
    this.command = new CommandHelper(page)
    this.bottomPanel = new BottomPanel(page)
    this.perf = new PerformanceHelper(page)
  }

  get visibleToasts() {
    return this.toast.visibleToasts
  }

  async setupUser(username: string) {
    const res = await this.request.get(`${this.url}/api/users`)
    if (res.status() !== 200)
      throw new Error(`Failed to retrieve users: ${await res.text()}`)

    const apiRes = await res.json()
    const user = Object.entries(apiRes?.users ?? {}).find(
      ([, name]) => name === username
    )
    const id = user?.[0]

    return id ? id : await this.createUser(username)
  }

  async createUser(username: string) {
    const resp = await this.request.post(`${this.url}/api/users`, {
      data: { username }
    })

    if (resp.status() !== 200)
      throw new Error(`Failed to create user: ${await resp.text()}`)

    return await resp.json()
  }

  async setupSettings(settings: Record<string, unknown>) {
    const resp = await this.request.post(
      `${this.url}/api/devtools/set_settings`,
      {
        data: settings
      }
    )

    if (resp.status() !== 200) {
      throw new Error(`Failed to setup settings: ${await resp.text()}`)
    }
  }

  async setup({
    clearStorage = true,
    mockReleases = true
  }: {
    clearStorage?: boolean
    mockReleases?: boolean
  } = {}) {
    await this.goto()

    // Mock release endpoint to prevent changelog popups
    if (mockReleases) {
      await this.page.route('**/releases**', async (route) => {
        const url = route.request().url()
        if (
          url.includes('api.comfy.org') ||
          url.includes('stagingapi.comfy.org')
        ) {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([])
          })
        } else {
          await route.continue()
        }
      })
    }

    if (clearStorage) {
      await this.page.evaluate((id) => {
        localStorage.clear()
        sessionStorage.clear()
        localStorage.setItem('Comfy.userId', id)
      }, this.id)
    }
    await this.goto()

    await this.page.waitForFunction(() => document.fonts.ready)
    await this.page.waitForFunction(
      () =>
        // window.app => GraphCanvas ready
        // window.app.extensionManager => GraphView ready
        window.app && window.app.extensionManager
    )
    await this.page.waitForSelector('.p-blockui-mask', { state: 'hidden' })
    await this.nextFrame()
  }

  public assetPath(fileName: string) {
    return `./browser_tests/assets/${fileName}`
  }

  async goto() {
    await this.page.goto(this.url)
  }

  async nextFrame() {
    await this.page.evaluate(() => {
      return new Promise<number>(requestAnimationFrame)
    })
  }

  async delay(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  /**
   * Attach a screenshot to the test report.
   * By default, screenshots are only taken in non-CI environments.
   * @param name - Name for the screenshot attachment
   * @param options - Optional configuration
   * @param options.runInCI - Whether to take screenshot in CI (default: false)
   * @param options.fullPage - Whether to capture full page (default: false)
   */
  async attachScreenshot(
    name: string,
    options: { runInCI?: boolean; fullPage?: boolean } = {}
  ) {
    const { runInCI = false, fullPage = false } = options

    // Skip in CI unless explicitly requested
    if (process.env.CI && !runInCI) {
      return
    }

    const testInfo = comfyPageFixture.info()
    await testInfo.attach(name, {
      body: await this.page.screenshot({ fullPage }),
      contentType: 'image/png'
    })
  }

  async closeMenu() {
    await this.page.click('button.comfy-close-menu-btn')
    await this.nextFrame()
  }

  async clickDialogButton(prompt: string, buttonText: string = 'Yes') {
    const modal = this.page.locator(
      `.comfy-modal-content:has-text("${prompt}")`
    )
    await modal.waitFor({ state: 'visible' })
    await modal
      .locator('.comfyui-button', {
        hasText: buttonText
      })
      .click()
    await modal.waitFor({ state: 'hidden' })
  }

  /** Get number of DOM widgets on the canvas. */
  async getDOMWidgetCount() {
    return await this.page.locator('.dom-widget').count()
  }

  async setFocusMode(focusMode: boolean) {
    await this.page.evaluate((focusMode) => {
      ;(window.app!.extensionManager as WorkspaceStore).focusMode = focusMode
    }, focusMode)
    await this.nextFrame()
  }
}

export const testComfySnapToGridGridSize = 50

export const comfyPageFixture = base.extend<{
  comfyPage: ComfyPage
  comfyMouse: ComfyMouse
}>({
  comfyPage: async ({ page, request }, use, testInfo) => {
    const comfyPage = new ComfyPage(page, request)

    const { parallelIndex } = testInfo
    const username = `playwright-test-${parallelIndex}`
    const userId = await comfyPage.setupUser(username)
    comfyPage.userIds[parallelIndex] = userId

    try {
      await comfyPage.setupSettings({
        'Comfy.UseNewMenu': 'Top',
        // Hide canvas menu/info/selection toolbox by default.
        'Comfy.Graph.CanvasInfo': false,
        'Comfy.Graph.CanvasMenu': false,
        'Comfy.Canvas.SelectionToolbox': false,
        // Hide all badges by default.
        'Comfy.NodeBadge.NodeIdBadgeMode': NodeBadgeMode.None,
        'Comfy.NodeBadge.NodeSourceBadgeMode': NodeBadgeMode.None,
        // Disable tooltips by default to avoid flakiness.
        'Comfy.EnableTooltips': false,
        'Comfy.userId': userId,
        // Set tutorial completed to true to avoid loading the tutorial workflow.
        'Comfy.TutorialCompleted': true,
        'Comfy.SnapToGrid.GridSize': testComfySnapToGridGridSize,
        'Comfy.VueNodes.AutoScaleLayout': false,
        // Disable toast warning about version compatibility, as they may or
        // may not appear - depending on upstream ComfyUI dependencies
        'Comfy.VersionCompatibility.DisableWarnings': true
      })
    } catch (e) {
      console.error(e)
    }

    await comfyPage.setup()

    const isPerf = testInfo.tags.includes('@perf')
    if (isPerf) await comfyPage.perf.init()

    await use(comfyPage)

    if (isPerf) await comfyPage.perf.dispose()
  },
  comfyMouse: async ({ comfyPage }, use) => {
    const comfyMouse = new ComfyMouse(comfyPage)
    await use(comfyMouse)
  }
})

const makeMatcher = function <T>(
  getValue: (node: NodeReference) => Promise<T> | T,
  type: string
) {
  return async function (
    this: ExpectMatcherState,
    node: NodeReference,
    options?: { timeout?: number; intervals?: number[] }
  ) {
    const value = await getValue(node)
    let assertion = expect(
      value,
      'Node is ' + (this.isNot ? '' : 'not ') + type
    )
    if (this.isNot) {
      assertion = assertion.not
    }
    await expect(async () => {
      assertion.toBeTruthy()
    }).toPass({ timeout: 250, ...options })
    return {
      pass: !this.isNot,
      message: () => 'Node is ' + (this.isNot ? 'not ' : '') + type
    }
  }
}

export const comfyExpect = expect.extend({
  toBePinned: makeMatcher((n) => n.isPinned(), 'pinned'),
  toBeBypassed: makeMatcher((n) => n.isBypassed(), 'bypassed'),
  toBeCollapsed: makeMatcher((n) => n.isCollapsed(), 'collapsed'),
  async toHaveFocus(locator: Locator, options = { timeout: 256 }) {
    const isFocused = await locator.evaluate(
      (el) => el === document.activeElement
    )

    await expect(async () => {
      expect(isFocused).toBe(!this.isNot)
    }).toPass(options)

    return {
      pass: isFocused,
      message: () => `Expected element to ${isFocused ? 'not ' : ''}be focused.`
    }
  }
})
