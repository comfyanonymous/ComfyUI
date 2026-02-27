import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { app } from '@/scripts/app'
import { useCommandStore } from '@/stores/commandStore'
import { useDialogStore } from '@/stores/dialogStore'

import { CORE_KEYBINDINGS } from './defaults'
import { KeyComboImpl } from './keyCombo'
import { KeybindingImpl } from './keybinding'
import { useKeybindingStore } from './keybindingStore'

export function useKeybindingService() {
  const keybindingStore = useKeybindingStore()
  const commandStore = useCommandStore()
  const settingStore = useSettingStore()
  const dialogStore = useDialogStore()

  function shouldForwardToCanvas(event: KeyboardEvent): boolean {
    if (event.ctrlKey || event.altKey || event.metaKey) {
      return false
    }

    const canvasKeys = ['Delete', 'Backspace']

    return canvasKeys.includes(event.key)
  }

  async function keybindHandler(event: KeyboardEvent) {
    const keyCombo = KeyComboImpl.fromEvent(event)
    if (keyCombo.isModifier) {
      return
    }

    const target = event.composedPath()[0] as HTMLElement
    if (
      keyCombo.isReservedByTextInput &&
      (target.tagName === 'TEXTAREA' ||
        target.tagName === 'INPUT' ||
        target.contentEditable === 'true' ||
        (target.tagName === 'SPAN' &&
          target.classList.contains('property_value')))
    ) {
      return
    }

    const keybinding = keybindingStore.getKeybinding(keyCombo)
    if (keybinding && keybinding.targetElementId !== 'graph-canvas') {
      if (
        event.key === 'Escape' &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.metaKey
      ) {
        if (dialogStore.dialogStack.length > 0) {
          return
        }
      }

      event.preventDefault()
      const runCommandIds = new Set([
        'Comfy.QueuePrompt',
        'Comfy.QueuePromptFront',
        'Comfy.QueueSelectedOutputNodes'
      ])
      if (runCommandIds.has(keybinding.commandId)) {
        await commandStore.execute(keybinding.commandId, {
          metadata: {
            trigger_source: 'keybinding'
          }
        })
      } else {
        await commandStore.execute(keybinding.commandId)
      }
      return
    }

    if (!keybinding && shouldForwardToCanvas(event)) {
      const canvas = app.canvas
      if (
        canvas &&
        canvas.processKey &&
        typeof canvas.processKey === 'function'
      ) {
        canvas.processKey(event)
        return
      }
    }

    if (event.ctrlKey || event.altKey || event.metaKey) {
      return
    }

    if (event.key === 'Escape') {
      const modals = document.querySelectorAll<HTMLElement>('.comfy-modal')
      for (const modal of modals) {
        const modalDisplay = window
          .getComputedStyle(modal)
          .getPropertyValue('display')

        if (modalDisplay !== 'none') {
          modal.style.display = 'none'
          break
        }
      }

      for (const d of document.querySelectorAll('dialog')) d.close()
    }
  }

  function registerCoreKeybindings() {
    for (const keybinding of CORE_KEYBINDINGS) {
      if (
        isCloud &&
        keybinding.commandId === 'Workspace.ToggleBottomPanelTab.logs-terminal'
      ) {
        continue
      }
      keybindingStore.addDefaultKeybinding(new KeybindingImpl(keybinding))
    }
  }

  function registerUserKeybindings() {
    const unsetBindings = settingStore.get('Comfy.Keybinding.UnsetBindings')
    for (const keybinding of unsetBindings) {
      keybindingStore.unsetKeybinding(new KeybindingImpl(keybinding))
    }
    const newBindings = settingStore.get('Comfy.Keybinding.NewBindings')
    for (const keybinding of newBindings) {
      if (
        isCloud &&
        keybinding.commandId === 'Workspace.ToggleBottomPanelTab.logs-terminal'
      ) {
        continue
      }
      keybindingStore.addUserKeybinding(new KeybindingImpl(keybinding))
    }
  }

  async function persistUserKeybindings() {
    await settingStore.setMany({
      'Comfy.Keybinding.NewBindings': Object.values(
        keybindingStore.getUserKeybindings()
      ),
      'Comfy.Keybinding.UnsetBindings': Object.values(
        keybindingStore.getUserUnsetKeybindings()
      )
    })
  }

  return {
    keybindHandler,
    registerCoreKeybindings,
    registerUserKeybindings,
    persistUserKeybindings
  }
}
