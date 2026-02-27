import { toRaw } from 'vue'

import { KeyComboImpl } from './keyCombo'
import type { Keybinding } from './types'

export class KeybindingImpl implements Keybinding {
  commandId: string
  combo: KeyComboImpl
  targetElementId?: string

  constructor(obj: Keybinding) {
    this.commandId = obj.commandId
    this.combo = new KeyComboImpl(obj.combo)
    this.targetElementId = obj.targetElementId
  }

  equals(other: unknown): boolean {
    const raw = toRaw(other)

    return raw instanceof KeybindingImpl
      ? this.commandId === raw.commandId &&
          this.combo.equals(raw.combo) &&
          this.targetElementId === raw.targetElementId
      : false
  }
}
