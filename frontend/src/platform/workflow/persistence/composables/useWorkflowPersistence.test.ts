import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import type * as I18n from 'vue-i18n'

import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { WorkflowDraftSnapshot } from '@/platform/workflow/persistence/base/draftCache'
import { useWorkflowPersistence } from '@/platform/workflow/persistence/composables/useWorkflowPersistence'
import { useWorkflowDraftStore } from '@/platform/workflow/persistence/stores/workflowDraftStore'
import { defaultGraphJSON } from '@/scripts/defaultGraph'
import { setStorageValue } from '@/scripts/utils'

const settingMocks = vi.hoisted(() => ({
  persistRef: null as { value: boolean } | null
}))

vi.mock('@/platform/settings/settingStore', async () => {
  const { ref } = await import('vue')
  settingMocks.persistRef = ref(true)
  return {
    useSettingStore: vi.fn(() => ({
      get: vi.fn((key: string) => {
        if (key === 'Comfy.Workflow.Persist')
          return settingMocks.persistRef!.value
        return undefined
      }),
      set: vi.fn()
    }))
  }
})

const mockToastAdd = vi.fn()
vi.mock('primevue', () => ({
  useToast: () => ({
    add: mockToastAdd
  })
}))

vi.mock('vue-i18n', async (importOriginal) => {
  const actual = await importOriginal<typeof I18n>()
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string) => key
    })
  }
})

const loadBlankWorkflow = vi.fn()
vi.mock('@/platform/workflow/core/services/workflowService', () => ({
  useWorkflowService: () => ({
    loadBlankWorkflow
  })
}))

vi.mock(
  '@/platform/workflow/templates/composables/useTemplateUrlLoader',
  () => ({
    useTemplateUrlLoader: () => ({
      loadTemplateFromUrlParams: vi.fn()
    })
  })
)

const executeCommand = vi.fn()
vi.mock('@/stores/commandStore', () => ({
  useCommandStore: () => ({
    execute: executeCommand
  })
}))

type GraphChangedHandler = (() => void) | null

const mocks = vi.hoisted(() => {
  const state = {
    graphChangedHandler: null as GraphChangedHandler,
    currentGraph: {} as Record<string, unknown>
  }
  const serializeMock = vi.fn(() => state.currentGraph)
  const loadGraphDataMock = vi.fn()
  const apiMock = {
    clientId: 'test-client',
    initialClientId: 'test-client',
    addEventListener: vi.fn((event: string, handler: () => void) => {
      if (event === 'graphChanged') {
        state.graphChangedHandler = handler
      }
    }),
    removeEventListener: vi.fn(),
    getUserData: vi.fn(),
    storeUserData: vi.fn(),
    listUserDataFullInfo: vi.fn(),
    storeSetting: vi.fn(),
    getSettings: vi.fn(),
    deleteUserData: vi.fn(),
    moveUserData: vi.fn(),
    apiURL: vi.fn((path: string) => path)
  }
  return { state, serializeMock, loadGraphDataMock, apiMock }
})

vi.mock('@/scripts/app', () => ({
  app: {
    graph: {
      serialize: () => mocks.serializeMock()
    },
    rootGraph: {
      serialize: () => mocks.serializeMock()
    },
    loadGraphData: (...args: unknown[]) => mocks.loadGraphDataMock(...args),
    canvas: {}
  }
}))

vi.mock('@/scripts/api', () => ({
  api: mocks.apiMock
}))

describe('useWorkflowPersistence', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2025-01-01T00:00:00Z'))
    setActivePinia(createTestingPinia({ stubActions: false }))
    localStorage.clear()
    sessionStorage.clear()
    vi.clearAllMocks()
    settingMocks.persistRef!.value = true
    mockToastAdd.mockClear()
    useWorkflowDraftStore().reset()
    mocks.state.graphChangedHandler = null
    mocks.state.currentGraph = { initial: true }
    mocks.serializeMock.mockImplementation(() => mocks.state.currentGraph)
    mocks.loadGraphDataMock.mockReset()
    mocks.apiMock.clientId = 'test-client'
    mocks.apiMock.initialClientId = 'test-client'
    mocks.apiMock.addEventListener.mockImplementation(
      (event: string, handler: () => void) => {
        if (event === 'graphChanged') {
          mocks.state.graphChangedHandler = handler
        }
      }
    )
    mocks.apiMock.removeEventListener.mockImplementation(() => {})
    mocks.apiMock.listUserDataFullInfo.mockResolvedValue([])
    mocks.apiMock.getUserData.mockResolvedValue({
      status: 200,
      text: () => Promise.resolve(defaultGraphJSON)
    } as Response)
    mocks.apiMock.apiURL.mockImplementation((path: string) => path)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('persists snapshots for multiple workflows', async () => {
    const workflowStore = useWorkflowStore()
    const workflowA = workflowStore.createTemporary('DraftA.json')
    await workflowStore.openWorkflow(workflowA)

    const persistence = useWorkflowPersistence()
    expect(persistence).toBeDefined()
    expect(mocks.state.graphChangedHandler).toBeTypeOf('function')

    const graphA = { title: 'A' }
    mocks.state.currentGraph = graphA
    mocks.state.graphChangedHandler!()
    await vi.advanceTimersByTimeAsync(800)

    const workflowB = workflowStore.createTemporary('DraftB.json')
    await workflowStore.openWorkflow(workflowB)
    const graphB = { title: 'B' }
    mocks.state.currentGraph = graphB
    mocks.state.graphChangedHandler!()
    await vi.advanceTimersByTimeAsync(800)

    const drafts = JSON.parse(
      localStorage.getItem('Comfy.Workflow.Drafts') ?? '{}'
    ) as Record<string, { data: string; isTemporary: boolean }>

    expect(Object.keys(drafts)).toEqual(
      expect.arrayContaining(['workflows/DraftA.json', 'workflows/DraftB.json'])
    )
    expect(JSON.parse(drafts['workflows/DraftA.json'].data)).toEqual(graphA)
    expect(JSON.parse(drafts['workflows/DraftB.json'].data)).toEqual(graphB)
    expect(drafts['workflows/DraftA.json'].isTemporary).toBe(true)
    expect(drafts['workflows/DraftB.json'].isTemporary).toBe(true)
  })

  it('evicts least recently used drafts beyond the limit', async () => {
    const workflowStore = useWorkflowStore()
    useWorkflowPersistence()
    expect(mocks.state.graphChangedHandler).toBeTypeOf('function')

    for (let i = 0; i < 33; i++) {
      const workflow = workflowStore.createTemporary(`Draft${i}.json`)
      await workflowStore.openWorkflow(workflow)
      mocks.state.currentGraph = { index: i }
      mocks.state.graphChangedHandler!()
      await vi.advanceTimersByTimeAsync(800)
      vi.setSystemTime(new Date(Date.now() + 60000))
    }

    const drafts = JSON.parse(
      localStorage.getItem('Comfy.Workflow.Drafts') ?? '{}'
    ) as Record<string, WorkflowDraftSnapshot>

    expect(Object.keys(drafts).length).toBe(32)
    expect(drafts['workflows/Draft0.json']).toBeUndefined()
    expect(drafts['workflows/Draft32.json']).toBeDefined()
  })

  it('restores temporary tabs from cached drafts', async () => {
    const workflowStore = useWorkflowStore()
    const draftStore = useWorkflowDraftStore()
    const draftData = JSON.parse(defaultGraphJSON)
    draftStore.saveDraft('workflows/Unsaved Workflow.json', {
      data: JSON.stringify(draftData),
      updatedAt: Date.now(),
      name: 'Unsaved Workflow.json',
      isTemporary: true
    })
    setStorageValue(
      'Comfy.OpenWorkflowsPaths',
      JSON.stringify(['workflows/Unsaved Workflow.json'])
    )
    setStorageValue('Comfy.ActiveWorkflowIndex', JSON.stringify(0))

    const { restoreWorkflowTabsState } = useWorkflowPersistence()
    restoreWorkflowTabsState()

    const restored = workflowStore.getWorkflowByPath(
      'workflows/Unsaved Workflow.json'
    )
    expect(restored).toBeTruthy()
    expect(restored?.isTemporary).toBe(true)
    expect(
      workflowStore.openWorkflows.map((workflow) => workflow?.path)
    ).toContain('workflows/Unsaved Workflow.json')
  })

  it('shows error toast when draft save fails', async () => {
    const workflowStore = useWorkflowStore()
    const draftStore = useWorkflowDraftStore()

    const workflow = workflowStore.createTemporary('FailingDraft.json')
    await workflowStore.openWorkflow(workflow)

    useWorkflowPersistence()
    expect(mocks.state.graphChangedHandler).toBeTypeOf('function')

    vi.spyOn(draftStore, 'saveDraft').mockImplementation(() => {
      throw new Error('Storage quota exceeded')
    })

    mocks.state.currentGraph = { title: 'Test' }
    mocks.state.graphChangedHandler!()
    await vi.advanceTimersByTimeAsync(800)

    expect(mockToastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        detail: expect.any(String)
      })
    )
  })

  it('clears all drafts when Persist is switched from true to false', async () => {
    const workflowStore = useWorkflowStore()
    const draftStore = useWorkflowDraftStore()
    const workflow = workflowStore.createTemporary('ClearDraft.json')
    await workflowStore.openWorkflow(workflow)

    useWorkflowPersistence()
    expect(mocks.state.graphChangedHandler).toBeTypeOf('function')

    mocks.state.currentGraph = { title: 'Draft to clear' }
    mocks.state.graphChangedHandler!()
    await vi.advanceTimersByTimeAsync(800)

    expect(draftStore.getDraft('workflows/ClearDraft.json')).toBeDefined()

    settingMocks.persistRef!.value = false
    await nextTick()

    expect(draftStore.getDraft('workflows/ClearDraft.json')).toBeUndefined()
  })
})
