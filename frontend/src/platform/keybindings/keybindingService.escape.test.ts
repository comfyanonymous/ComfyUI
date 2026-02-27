import { createPinia, setActivePinia } from 'pinia'
import { markRaw, reactive } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { CORE_KEYBINDINGS } from '@/platform/keybindings/defaults'
import { KeyComboImpl } from '@/platform/keybindings/keyCombo'
import { KeybindingImpl } from '@/platform/keybindings/keybinding'
import { useKeybindingService } from '@/platform/keybindings/keybindingService'
import { useKeybindingStore } from '@/platform/keybindings/keybindingStore'
import { useCommandStore } from '@/stores/commandStore'
import type { DialogInstance } from '@/stores/dialogStore'
import { useDialogStore } from '@/stores/dialogStore'

function createTestDialogInstance(
  key: string,
  overrides: Partial<DialogInstance> = {}
): DialogInstance {
  return {
    key,
    visible: true,
    component: markRaw({ template: '<div />' }),
    contentProps: {},
    dialogComponentProps: {},
    priority: 0,
    ...overrides
  }
}

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn(() => ({
    get: vi.fn(() => [])
  }))
}))

vi.mock('@/stores/dialogStore', () => {
  const dialogStack = reactive<DialogInstance[]>([])
  return {
    useDialogStore: () => ({ dialogStack })
  }
})

vi.mock('@/scripts/app', () => ({
  app: {
    canvas: null
  }
}))

describe('keybindingService - Escape key handling', () => {
  let keybindingService: ReturnType<typeof useKeybindingService>
  let mockCommandExecute: ReturnType<typeof useCommandStore>['execute']

  beforeEach(() => {
    vi.clearAllMocks()
    setActivePinia(createPinia())

    const commandStore = useCommandStore()
    mockCommandExecute = vi.fn()
    commandStore.execute = mockCommandExecute

    const dialogStore = useDialogStore()
    dialogStore.dialogStack.length = 0

    keybindingService = useKeybindingService()
    keybindingService.registerCoreKeybindings()
  })

  function createKeyboardEvent(
    key: string,
    options: {
      ctrlKey?: boolean
      altKey?: boolean
      metaKey?: boolean
      shiftKey?: boolean
    } = {}
  ): KeyboardEvent {
    const event = new KeyboardEvent('keydown', {
      key,
      ctrlKey: options.ctrlKey ?? false,
      altKey: options.altKey ?? false,
      metaKey: options.metaKey ?? false,
      shiftKey: options.shiftKey ?? false,
      bubbles: true,
      cancelable: true
    })

    event.preventDefault = vi.fn()
    event.composedPath = vi.fn(() => [document.body])

    return event
  }

  it('should execute Escape keybinding when no dialogs are open', async () => {
    const event = createKeyboardEvent('Escape')
    await keybindingService.keybindHandler(event)

    expect(mockCommandExecute).toHaveBeenCalledWith('Comfy.Graph.ExitSubgraph')
  })

  it('should NOT execute Escape keybinding when dialogs are open', async () => {
    const dialogStore = useDialogStore()
    dialogStore.dialogStack.push(createTestDialogInstance('test-dialog'))

    keybindingService = useKeybindingService()

    const event = createKeyboardEvent('Escape')
    await keybindingService.keybindHandler(event)

    expect(mockCommandExecute).not.toHaveBeenCalled()
  })

  it('should execute Escape keybinding with modifiers regardless of dialog state', async () => {
    const dialogStore = useDialogStore()
    dialogStore.dialogStack.push(createTestDialogInstance('test-dialog'))

    const keybindingStore = useKeybindingStore()
    keybindingStore.addDefaultKeybinding(
      new KeybindingImpl({
        commandId: 'Test.CtrlEscape',
        combo: { key: 'Escape', ctrl: true }
      })
    )

    keybindingService = useKeybindingService()

    const event = createKeyboardEvent('Escape', { ctrlKey: true })
    await keybindingService.keybindHandler(event)

    expect(mockCommandExecute).toHaveBeenCalledWith('Test.CtrlEscape')
  })

  it('should verify Escape keybinding exists in CORE_KEYBINDINGS', () => {
    const escapeBinding = CORE_KEYBINDINGS.find(
      (kb) => kb.combo.key === 'Escape' && !kb.combo.ctrl && !kb.combo.alt
    )

    expect(escapeBinding).toBeDefined()
    expect(escapeBinding?.commandId).toBe('Comfy.Graph.ExitSubgraph')
  })

  it('should create correct KeyComboImpl from Escape event', () => {
    const event = new KeyboardEvent('keydown', {
      key: 'Escape',
      ctrlKey: false,
      altKey: false,
      metaKey: false,
      shiftKey: false
    })

    const keyCombo = KeyComboImpl.fromEvent(event)

    expect(keyCombo.key).toBe('Escape')
    expect(keyCombo.ctrl).toBe(false)
    expect(keyCombo.alt).toBe(false)
    expect(keyCombo.shift).toBe(false)
  })

  it('should still close legacy modals on Escape when no keybinding matched', async () => {
    setActivePinia(createPinia())
    keybindingService = useKeybindingService()

    const mockModal = document.createElement('div')
    mockModal.className = 'comfy-modal'
    mockModal.style.display = 'block'
    document.body.appendChild(mockModal)

    const originalGetComputedStyle = window.getComputedStyle
    window.getComputedStyle = vi.fn().mockReturnValue({
      getPropertyValue: vi.fn().mockReturnValue('block')
    })

    try {
      const event = createKeyboardEvent('Escape')
      await keybindingService.keybindHandler(event)

      expect(mockModal.style.display).toBe('none')
    } finally {
      document.body.removeChild(mockModal)
      window.getComputedStyle = originalGetComputedStyle
    }
  })
})
