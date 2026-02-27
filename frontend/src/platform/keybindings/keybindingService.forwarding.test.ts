import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useKeybindingService } from '@/platform/keybindings/keybindingService'
import { app } from '@/scripts/app'
import { useCommandStore } from '@/stores/commandStore'
import { useDialogStore } from '@/stores/dialogStore'

vi.mock('@/scripts/app', () => {
  return {
    app: {
      canvas: {
        processKey: vi.fn()
      }
    }
  }
})

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn(() => [])
  }))
}))

vi.mock('@/stores/dialogStore', () => ({
  useDialogStore: vi.fn(() => ({
    dialogStack: []
  }))
}))

function createTestKeyboardEvent(
  key: string,
  options: {
    target?: Element
    ctrlKey?: boolean
    altKey?: boolean
    metaKey?: boolean
  } = {}
): KeyboardEvent {
  const {
    target = document.body,
    ctrlKey = false,
    altKey = false,
    metaKey = false
  } = options

  const event = new KeyboardEvent('keydown', {
    key,
    ctrlKey,
    altKey,
    metaKey,
    bubbles: true,
    cancelable: true
  })

  event.preventDefault = vi.fn()
  event.composedPath = vi.fn(() => [target])

  return event
}

describe('keybindingService - Event Forwarding', () => {
  let keybindingService: ReturnType<typeof useKeybindingService>

  beforeEach(() => {
    vi.clearAllMocks()
    setActivePinia(createTestingPinia({ stubActions: false }))

    const commandStore = useCommandStore()
    commandStore.execute = vi.fn()

    vi.mocked(useDialogStore).mockReturnValue({
      dialogStack: []
    } as Partial<ReturnType<typeof useDialogStore>> as ReturnType<
      typeof useDialogStore
    >)

    keybindingService = useKeybindingService()
    keybindingService.registerCoreKeybindings()
  })

  it('should forward Delete key to canvas when no keybinding exists', async () => {
    const event = createTestKeyboardEvent('Delete')

    await keybindingService.keybindHandler(event)

    expect(vi.mocked(app.canvas.processKey)).toHaveBeenCalledWith(event)
    expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
  })

  it('should forward Backspace key to canvas when no keybinding exists', async () => {
    const event = createTestKeyboardEvent('Backspace')

    await keybindingService.keybindHandler(event)

    expect(vi.mocked(app.canvas.processKey)).toHaveBeenCalledWith(event)
    expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
  })

  it('should not forward Delete key when typing in input field', async () => {
    const inputElement = document.createElement('input')
    const event = createTestKeyboardEvent('Delete', { target: inputElement })

    await keybindingService.keybindHandler(event)

    expect(vi.mocked(app.canvas.processKey)).not.toHaveBeenCalled()
    expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
  })

  it('should not forward Delete key when typing in textarea', async () => {
    const textareaElement = document.createElement('textarea')
    const event = createTestKeyboardEvent('Delete', { target: textareaElement })

    await keybindingService.keybindHandler(event)

    expect(vi.mocked(app.canvas.processKey)).not.toHaveBeenCalled()
    expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
  })

  it('should not forward Delete key when canvas processKey is not available', async () => {
    // Temporarily replace processKey with undefined - testing edge case
    const originalProcessKey = vi.mocked(app.canvas).processKey
    vi.mocked(app.canvas).processKey = undefined!

    const event = createTestKeyboardEvent('Delete')

    try {
      await keybindingService.keybindHandler(event)
      expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
    } finally {
      // Restore processKey for other tests
      vi.mocked(app.canvas).processKey = originalProcessKey
    }
  })

  it('should not forward Delete key when canvas is not available', async () => {
    const originalCanvas = vi.mocked(app).canvas
    vi.mocked(app).canvas = null!

    const event = createTestKeyboardEvent('Delete')

    try {
      await keybindingService.keybindHandler(event)
      expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
    } finally {
      // Restore canvas for other tests
      vi.mocked(app).canvas = originalCanvas
    }
  })

  it('should not forward non-canvas keys', async () => {
    const event = createTestKeyboardEvent('Enter')

    await keybindingService.keybindHandler(event)

    expect(vi.mocked(app.canvas.processKey)).not.toHaveBeenCalled()
    expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
  })

  it('should not forward when modifier keys are pressed', async () => {
    const event = createTestKeyboardEvent('Delete', { ctrlKey: true })

    await keybindingService.keybindHandler(event)

    expect(vi.mocked(app.canvas.processKey)).not.toHaveBeenCalled()
    expect(vi.mocked(useCommandStore().execute)).not.toHaveBeenCalled()
  })
})
