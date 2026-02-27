import { createPinia, setActivePinia } from 'pinia'
import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWorkflowActionsMenu } from '@/composables/useWorkflowActionsMenu'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import type { WorkflowMenuAction } from '@/types/workflowMenuItem'

vi.mock('vue-i18n', () => ({
  useI18n: vi.fn(() => ({
    t: (key: string) => key
  }))
}))

const mockBookmarkStore = vi.hoisted(() => ({
  isBookmarked: vi.fn(() => false),
  toggleBookmarked: vi.fn()
}))

const mockWorkflowStore = vi.hoisted(() => ({
  activeWorkflow: { path: 'test.json', isPersisted: true } as ComfyWorkflow
}))

const mockWorkflowService = vi.hoisted(() => ({
  openWorkflow: vi.fn(),
  duplicateWorkflow: vi.fn(),
  saveWorkflowAs: vi.fn(),
  deleteWorkflow: vi.fn()
}))

const mockCommandStore = vi.hoisted(() => ({
  execute: vi.fn()
}))

const mockSubgraphStore = vi.hoisted(() => ({
  isSubgraphBlueprint: vi.fn(() => false)
}))

const mockMenuItemStore = vi.hoisted(() => ({
  hasSeenLinear: false
}))

const mockCanvasStore = vi.hoisted(() => ({
  linearMode: false
}))

const mockFeatureFlags = vi.hoisted(() => ({
  flags: { linearToggleEnabled: false }
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: vi.fn(() => mockWorkflowStore),
  useWorkflowBookmarkStore: vi.fn(() => mockBookmarkStore)
}))

vi.mock('@/platform/workflow/core/services/workflowService', () => ({
  useWorkflowService: vi.fn(() => mockWorkflowService)
}))

vi.mock('@/stores/commandStore', () => ({
  useCommandStore: vi.fn(() => mockCommandStore)
}))

vi.mock('@/stores/subgraphStore', () => ({
  useSubgraphStore: vi.fn(() => mockSubgraphStore)
}))

vi.mock('@/stores/menuItemStore', () => ({
  useMenuItemStore: vi.fn(() => mockMenuItemStore)
}))

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: vi.fn(() => mockCanvasStore)
}))

vi.mock('@/composables/useFeatureFlags', () => ({
  useFeatureFlags: vi.fn(() => mockFeatureFlags)
}))

type MenuItems = ReturnType<typeof useWorkflowActionsMenu>['menuItems']['value']

function actionItems(items: MenuItems): WorkflowMenuAction[] {
  return items.filter((i): i is WorkflowMenuAction => !i.separator)
}

function menuLabels(items: MenuItems) {
  return actionItems(items).map((i) => i.label)
}

function findItem(items: MenuItems, label: string): WorkflowMenuAction {
  const item = actionItems(items).find((i) => i.label === label)
  if (!item) throw new Error(`Menu item "${label}" not found`)
  return item
}

describe('useWorkflowActionsMenu', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mockBookmarkStore.isBookmarked.mockReturnValue(false)
    mockSubgraphStore.isSubgraphBlueprint.mockReturnValue(false)
    mockMenuItemStore.hasSeenLinear = false
    mockCanvasStore.linearMode = false
    mockFeatureFlags.flags.linearToggleEnabled = false
    mockWorkflowStore.activeWorkflow = {
      path: 'test.json',
      isPersisted: true
    } as ComfyWorkflow
  })

  it('shows root-level items by default', () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('g.rename')
    expect(labels).toContain('breadcrumbsMenu.duplicate')
    expect(labels).toContain('menuLabels.Save')
    expect(labels).toContain('menuLabels.Save As')
    expect(labels).toContain('menuLabels.Export')
    expect(labels).toContain('menuLabels.Export (API)')
    expect(labels).toContain('breadcrumbsMenu.clearWorkflow')
    expect(labels).toContain('breadcrumbsMenu.deleteWorkflow')
  })

  it('hides root-only items when isRoot is false', () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: false })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('g.rename')
    expect(labels).toContain('breadcrumbsMenu.clearWorkflow')
    expect(labels).not.toContain('breadcrumbsMenu.duplicate')
    expect(labels).not.toContain('menuLabels.Save')
    expect(labels).not.toContain('menuLabels.Save As')
  })

  it('hides delete item when includeDelete is false', () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), {
      isRoot: true,
      includeDelete: false
    })
    const labels = menuLabels(menuItems.value)

    expect(labels).not.toContain('breadcrumbsMenu.deleteWorkflow')
  })

  it('shows app mode items when linearToggleEnabled flag is set', () => {
    mockFeatureFlags.flags.linearToggleEnabled = true

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('breadcrumbsMenu.enterAppMode')
  })

  it('shows app mode items when user has seen linear mode', () => {
    mockMenuItemStore.hasSeenLinear = true

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('breadcrumbsMenu.enterAppMode')
  })

  it('hides app mode items when conditions not met', () => {
    mockMenuItemStore.hasSeenLinear = false
    mockFeatureFlags.flags.linearToggleEnabled = false

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).not.toContain('breadcrumbsMenu.enterAppMode')
  })

  it('hides app mode items when not root', () => {
    mockFeatureFlags.flags.linearToggleEnabled = true

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: false })
    const labels = menuLabels(menuItems.value)

    expect(labels).not.toContain('breadcrumbsMenu.enterAppMode')
  })

  it('shows "go to workflow mode" when in linear mode', () => {
    mockFeatureFlags.flags.linearToggleEnabled = true
    mockCanvasStore.linearMode = true

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('breadcrumbsMenu.exitAppMode')
    expect(labels).not.toContain('breadcrumbsMenu.enterAppMode')
  })

  it('shows bookmark label based on bookmark state', () => {
    mockBookmarkStore.isBookmarked.mockReturnValue(true)

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('tabMenu.removeFromBookmarks')
    expect(labels).not.toContain('tabMenu.addToBookmarks')
  })

  it('adds badge to app mode items', () => {
    mockFeatureFlags.flags.linearToggleEnabled = true

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const appModeItem = findItem(
      menuItems.value,
      'breadcrumbsMenu.enterAppMode'
    )

    expect(appModeItem.badge).toBeDefined()
  })

  it('calls startRename when rename command is invoked', async () => {
    const startRename = vi.fn()
    const { menuItems } = useWorkflowActionsMenu(startRename, {
      isRoot: true
    })

    await findItem(menuItems.value, 'g.rename').command?.()

    expect(startRename).toHaveBeenCalled()
  })

  it('uses provided workflow ref instead of activeWorkflow', () => {
    const customWorkflow = ref({
      path: 'custom.json',
      isPersisted: true,
      isTemporary: false
    } as ComfyWorkflow)

    mockBookmarkStore.isBookmarked.mockReturnValue(false)

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), {
      isRoot: true,
      workflow: customWorkflow
    })

    expect(menuItems.value.length).toBeGreaterThan(0)
    expect(mockBookmarkStore.isBookmarked).toHaveBeenCalledWith('custom.json')
  })

  it('shows publish item for blueprints', () => {
    mockSubgraphStore.isSubgraphBlueprint.mockReturnValue(true)

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const labels = menuLabels(menuItems.value)

    expect(labels).toContain('subgraphStore.publish')
    expect(labels).toContain('breadcrumbsMenu.deleteBlueprint')
    expect(labels).not.toContain('breadcrumbsMenu.duplicate')
  })

  it('duplicate command calls workflowService.duplicateWorkflow', async () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    await findItem(menuItems.value, 'breadcrumbsMenu.duplicate').command?.()

    expect(mockWorkflowService.duplicateWorkflow).toHaveBeenCalledWith(
      mockWorkflowStore.activeWorkflow
    )
  })

  it('save command executes Comfy.SaveWorkflow', async () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    await findItem(menuItems.value, 'menuLabels.Save').command?.()

    expect(mockCommandStore.execute).toHaveBeenCalledWith('Comfy.SaveWorkflow')
  })

  it('delete command calls workflowService.deleteWorkflow', async () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    await findItem(
      menuItems.value,
      'breadcrumbsMenu.deleteWorkflow'
    ).command?.()

    expect(mockWorkflowService.deleteWorkflow).toHaveBeenCalledWith(
      mockWorkflowStore.activeWorkflow
    )
  })

  it('bookmark toggle calls bookmarkStore.toggleBookmarked', async () => {
    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    await findItem(menuItems.value, 'tabMenu.addToBookmarks').command?.()

    expect(mockBookmarkStore.toggleBookmarked).toHaveBeenCalledWith('test.json')
  })

  it('app mode toggle executes Comfy.ToggleLinear', async () => {
    mockFeatureFlags.flags.linearToggleEnabled = true

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    await findItem(menuItems.value, 'breadcrumbsMenu.enterAppMode').command?.()

    expect(mockCommandStore.execute).toHaveBeenCalledWith(
      'Comfy.ToggleLinear',
      { metadata: { source: 'breadcrumb_menu' } }
    )
  })

  it('rename is disabled for unpersisted root workflows', () => {
    mockWorkflowStore.activeWorkflow = {
      path: 'test.json',
      isPersisted: false
    } as ComfyWorkflow

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const rename = findItem(menuItems.value, 'g.rename')

    expect(rename.disabled).toBe(true)
  })

  it('bookmark is disabled for temporary workflows', () => {
    mockWorkflowStore.activeWorkflow = {
      path: 'test.json',
      isPersisted: true,
      isTemporary: true
    } as ComfyWorkflow

    const { menuItems } = useWorkflowActionsMenu(vi.fn(), { isRoot: true })
    const bookmark = findItem(menuItems.value, 'tabMenu.addToBookmarks')

    expect(bookmark.disabled).toBe(true)
  })

  it('switches to custom workflow before executing rename', async () => {
    const customWorkflow = ref({
      path: 'other.json',
      isPersisted: true
    } as ComfyWorkflow)
    const startRename = vi.fn()

    const { menuItems } = useWorkflowActionsMenu(startRename, {
      isRoot: true,
      workflow: customWorkflow
    })
    await findItem(menuItems.value, 'g.rename').command?.()

    expect(mockWorkflowService.openWorkflow).toHaveBeenCalledWith(
      customWorkflow.value
    )
    expect(startRename).toHaveBeenCalled()
  })
})
