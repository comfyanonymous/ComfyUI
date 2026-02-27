import { beforeEach, describe, expect, it, vi } from 'vitest'
import { effectScope, nextTick, reactive } from 'vue'
import type { EffectScope } from 'vue'

import { useBrowserTabTitle } from '@/composables/useBrowserTabTitle'

// Mock i18n module
vi.mock('@/i18n', () => ({
  t: (key: string, fallback: string) =>
    key === 'g.nodesRunning' ? 'nodes running' : fallback
}))

// Mock the execution store
const executionStore = reactive<{
  isIdle: boolean
  executionProgress: number
  executingNode: unknown
  executingNodeProgress: number
  nodeProgressStates: Record<string, unknown>
  activeJob: {
    workflow: {
      changeTracker: {
        activeState: {
          nodes: { id: number; type: string }[]
        }
      }
    }
  } | null
}>({
  isIdle: true,
  executionProgress: 0,
  executingNode: null,
  executingNodeProgress: 0,
  nodeProgressStates: {},
  activeJob: null
})
vi.mock('@/stores/executionStore', () => ({
  useExecutionStore: () => executionStore
}))

// Mock the setting store
const settingStore = reactive({
  get: vi.fn((_key: string) => 'Enabled')
})
vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => settingStore
}))

// Mock the workflow store
const workflowStore = reactive<{
  activeWorkflow: {
    filename: string
    isModified: boolean
    isPersisted: boolean
  } | null
}>({
  activeWorkflow: null
})
vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: () => workflowStore
}))

// Mock the workspace store
const workspaceStore = reactive({
  shiftDown: false
})
vi.mock('@/stores/workspaceStore', () => ({
  useWorkspaceStore: () => workspaceStore
}))

describe('useBrowserTabTitle', () => {
  beforeEach(() => {
    // reset execution store
    executionStore.isIdle = true
    executionStore.executionProgress = 0
    executionStore.executingNode = null
    executionStore.executingNodeProgress = 0
    executionStore.nodeProgressStates = {}
    executionStore.activeJob = null

    // reset setting and workflow stores
    vi.mocked(settingStore.get).mockReturnValue('Enabled')
    workflowStore.activeWorkflow = null
    workspaceStore.shiftDown = false

    // reset document title
    document.title = ''
  })

  it('sets default title when idle and no workflow', () => {
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    expect(document.title).toBe('ComfyUI')
    scope.stop()
  })

  it('sets workflow name as title when workflow exists and menu enabled', async () => {
    vi.mocked(settingStore.get).mockReturnValue('Enabled')
    workflowStore.activeWorkflow = {
      filename: 'myFlow',
      isModified: false,
      isPersisted: true
    }
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    await nextTick()
    expect(document.title).toBe('myFlow - ComfyUI')
    scope.stop()
  })

  it('adds asterisk for unsaved workflow', async () => {
    vi.mocked(settingStore.get).mockReturnValue('Enabled')
    workflowStore.activeWorkflow = {
      filename: 'myFlow',
      isModified: true,
      isPersisted: true
    }
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    await nextTick()
    expect(document.title).toBe('*myFlow - ComfyUI')
    scope.stop()
  })

  it('hides asterisk when autosave is enabled', async () => {
    vi.mocked(settingStore.get).mockImplementation((key: string) => {
      if (key === 'Comfy.Workflow.AutoSave') return 'after delay'
      if (key === 'Comfy.UseNewMenu') return 'Enabled'
      return 'Enabled'
    })
    workflowStore.activeWorkflow = {
      filename: 'myFlow',
      isModified: true,
      isPersisted: true
    }
    useBrowserTabTitle()
    await nextTick()
    expect(document.title).toBe('myFlow - ComfyUI')
  })

  it('hides asterisk while Shift key is held', async () => {
    vi.mocked(settingStore.get).mockImplementation((key: string) => {
      if (key === 'Comfy.Workflow.AutoSave') return 'off'
      if (key === 'Comfy.UseNewMenu') return 'Enabled'
      return 'Enabled'
    })
    workspaceStore.shiftDown = true
    workflowStore.activeWorkflow = {
      filename: 'myFlow',
      isModified: true,
      isPersisted: true
    }
    useBrowserTabTitle()
    await nextTick()
    expect(document.title).toBe('myFlow - ComfyUI')
  })

  // Fails when run together with other tests. Suspect to be caused by leaked
  // state from previous tests.
  it.skip('disables workflow title when menu disabled', async () => {
    vi.mocked(settingStore.get).mockReturnValue('Disabled')
    workflowStore.activeWorkflow = {
      filename: 'myFlow',
      isModified: false,
      isPersisted: true
    }
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    await nextTick()
    expect(document.title).toBe('ComfyUI')
    scope.stop()
  })

  it('shows execution progress when not idle without workflow', async () => {
    executionStore.isIdle = false
    executionStore.executionProgress = 0.3
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    await nextTick()
    expect(document.title).toBe('[30%]ComfyUI')
    scope.stop()
  })

  it('shows node execution title when executing a node using nodeProgressStates', async () => {
    executionStore.isIdle = false
    executionStore.executionProgress = 0.4
    executionStore.nodeProgressStates = {
      '1': { state: 'running', value: 5, max: 10, node: '1', prompt_id: 'test' }
    }
    executionStore.activeJob = {
      workflow: {
        changeTracker: {
          activeState: {
            nodes: [{ id: 1, type: 'Foo' }]
          }
        }
      }
    }
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    await nextTick()
    expect(document.title).toBe('[40%][50%] Foo')
    scope.stop()
  })

  it('shows multiple nodes running when multiple nodes are executing', async () => {
    executionStore.isIdle = false
    executionStore.executionProgress = 0.4
    executionStore.nodeProgressStates = {
      '1': {
        state: 'running',
        value: 5,
        max: 10,
        node: '1',
        prompt_id: 'test'
      },
      '2': { state: 'running', value: 8, max: 10, node: '2', prompt_id: 'test' }
    }
    const scope: EffectScope = effectScope()
    scope.run(() => useBrowserTabTitle())
    await nextTick()
    expect(document.title).toBe('[40%][2 nodes running]')
    scope.stop()
  })
})
