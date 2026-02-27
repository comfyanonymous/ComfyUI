import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import { useWorkflowAutoSave } from '@/platform/workflow/persistence/composables/useWorkflowAutoSave'
import { api } from '@/scripts/api'

vi.mock('@/scripts/api', () => ({
  api: {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }
}))

vi.mock('@/platform/workflow/core/services/workflowService', () => ({
  useWorkflowService: vi.fn(() => ({
    saveWorkflow: vi.fn()
  }))
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn((key) => {
      if (key === 'Comfy.Workflow.AutoSave') return mockAutoSaveSetting
      if (key === 'Comfy.Workflow.AutoSaveDelay') return mockAutoSaveDelay
      return null
    })
  }))
}))

vi.mock('@/platform/workflow/management/stores/workflowStore', () => ({
  useWorkflowStore: vi.fn(() => ({
    activeWorkflow: mockActiveWorkflow
  }))
}))

let mockAutoSaveSetting: string = 'off'
let mockAutoSaveDelay: number = 1000
let mockActiveWorkflow: { isModified: boolean; isPersisted?: boolean } | null =
  null

describe('useWorkflowAutoSave', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('should auto-save workflow after delay when modified and autosave enabled', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 1000
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    vi.advanceTimersByTime(1000)

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    expect(serviceInstance.saveWorkflow).toHaveBeenCalledWith(
      mockActiveWorkflow
    )
  })

  it('should not auto-save workflow after delay when not modified and autosave enabled', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 1000
    mockActiveWorkflow = { isModified: false, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    vi.advanceTimersByTime(1000)

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    expect(serviceInstance.saveWorkflow).not.toHaveBeenCalledWith(
      mockActiveWorkflow
    )
  })

  it('should not auto save workflow when autosave is off', async () => {
    mockAutoSaveSetting = 'off'
    mockAutoSaveDelay = 1000
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    vi.advanceTimersByTime(mockAutoSaveDelay)

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    expect(serviceInstance.saveWorkflow).not.toHaveBeenCalled()
  })

  it('should respect the user specified auto save delay', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 2000
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    vi.advanceTimersByTime(1000)

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    expect(serviceInstance.saveWorkflow).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1000)

    expect(serviceInstance.saveWorkflow).toHaveBeenCalled()
  })

  it('should debounce save requests', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 2000
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    const graphChangedCallback = vi.mocked(api.addEventListener).mock
      .calls[0][1]

    graphChangedCallback?.({} as Parameters<typeof graphChangedCallback>[0])

    vi.advanceTimersByTime(500)

    graphChangedCallback?.({} as Parameters<typeof graphChangedCallback>[0])

    vi.advanceTimersByTime(1999)
    expect(serviceInstance.saveWorkflow).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1)
    expect(serviceInstance.saveWorkflow).toHaveBeenCalledTimes(1)
  })

  it('should handle save error gracefully', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 1000
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    const consoleErrorSpy = vi
      .spyOn(console, 'error')
      .mockImplementation(() => {})

    try {
      mount({
        template: `<div></div>`,
        setup() {
          useWorkflowAutoSave()
          return {}
        }
      })

      const serviceInstance =
        vi.mocked(useWorkflowService).mock.results[0].value
      serviceInstance.saveWorkflow.mockRejectedValue(new Error('Test Error'))

      vi.advanceTimersByTime(1000)
      await Promise.resolve()

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Auto save failed:',
        expect.any(Error)
      )
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it('should queue autosave requests during saving and reschedule after save completes', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 1000
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    let resolveSave: () => void
    const firstSavePromise = new Promise<void>((resolve) => {
      resolveSave = resolve
    })

    serviceInstance.saveWorkflow.mockImplementationOnce(() => firstSavePromise)

    vi.advanceTimersByTime(1000)

    const graphChangedCallback = vi.mocked(api.addEventListener).mock
      .calls[0][1]
    graphChangedCallback?.({} as Parameters<typeof graphChangedCallback>[0])

    resolveSave!()
    await Promise.resolve()

    vi.advanceTimersByTime(1000)
    expect(serviceInstance.saveWorkflow).toHaveBeenCalledTimes(2)
  })

  it('should clean up event listeners on component unmount', async () => {
    mockAutoSaveSetting = 'after delay'

    const wrapper = mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    wrapper.unmount()

    expect(api.removeEventListener).toHaveBeenCalled()
  })

  it('should handle edge case delay values properly', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 0
    mockActiveWorkflow = { isModified: true, isPersisted: true }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    await vi.runAllTimersAsync()

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    expect(serviceInstance.saveWorkflow).toHaveBeenCalledTimes(1)
    serviceInstance.saveWorkflow.mockClear()

    mockAutoSaveDelay = -500

    const graphChangedCallback = vi.mocked(api.addEventListener).mock
      .calls[0][1]
    graphChangedCallback?.({} as Parameters<typeof graphChangedCallback>[0])

    await vi.runAllTimersAsync()

    expect(serviceInstance.saveWorkflow).toHaveBeenCalledTimes(1)
  })

  it('should not autosave if workflow is not persisted', async () => {
    mockAutoSaveSetting = 'after delay'
    mockAutoSaveDelay = 1000
    mockActiveWorkflow = { isModified: true, isPersisted: false }

    mount({
      template: `<div></div>`,
      setup() {
        useWorkflowAutoSave()
        return {}
      }
    })

    vi.advanceTimersByTime(1000)

    const serviceInstance = vi.mocked(useWorkflowService).mock.results[0].value
    expect(serviceInstance.saveWorkflow).not.toHaveBeenCalledWith(
      mockActiveWorkflow
    )
  })
})
