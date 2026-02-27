import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { KeybindingImpl } from '@/platform/keybindings/keybinding'
import { useKeybindingStore } from '@/platform/keybindings/keybindingStore'

describe('useKeybindingStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  it('should add and retrieve default keybindings', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'A', ctrl: true }
    })

    store.addDefaultKeybinding(keybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(keybinding.combo)).toEqual(keybinding)
  })

  it('should add and retrieve user keybindings', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'B', alt: true }
    })

    store.addUserKeybinding(keybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(keybinding.combo)).toEqual(keybinding)
  })

  it('should get keybindings by command id', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'C', ctrl: true }
    })
    store.addDefaultKeybinding(keybinding)
    expect(store.getKeybindingsByCommandId('test.command')).toEqual([
      keybinding
    ])
  })

  it('should override default keybindings with user keybindings', () => {
    const store = useKeybindingStore()
    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command1',
      combo: { key: 'C', ctrl: true }
    })
    const userKeybinding = new KeybindingImpl({
      commandId: 'test.command2',
      combo: { key: 'C', ctrl: true }
    })

    store.addDefaultKeybinding(defaultKeybinding)
    store.addUserKeybinding(userKeybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(userKeybinding.combo)).toEqual(userKeybinding)
  })

  it('Should allow binding to unsetted default keybindings', () => {
    const store = useKeybindingStore()
    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command1',
      combo: { key: 'C', ctrl: true }
    })
    store.addDefaultKeybinding(defaultKeybinding)
    store.unsetKeybinding(defaultKeybinding)

    const userKeybinding = new KeybindingImpl({
      commandId: 'test.command2',
      combo: { key: 'C', ctrl: true }
    })
    store.addUserKeybinding(userKeybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(userKeybinding.combo)).toEqual(userKeybinding)
  })

  it('should unset user keybindings', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'D', meta: true }
    })

    store.addUserKeybinding(keybinding)
    expect(store.keybindings).toHaveLength(1)

    store.unsetKeybinding(keybinding)
    expect(store.keybindings).toHaveLength(0)
  })

  it('should unset default keybindings', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'E', ctrl: true, alt: true }
    })

    store.addDefaultKeybinding(keybinding)
    expect(store.keybindings).toHaveLength(1)

    store.unsetKeybinding(keybinding)
    expect(store.keybindings).toHaveLength(0)
  })

  it('should throw an error when adding duplicate default keybindings', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'F', shift: true }
    })

    store.addDefaultKeybinding(keybinding)
    expect(() => store.addDefaultKeybinding(keybinding)).toThrow()
  })

  it('should allow adding duplicate user keybindings', () => {
    const store = useKeybindingStore()
    const keybinding1 = new KeybindingImpl({
      commandId: 'test.command1',
      combo: { key: 'G', ctrl: true }
    })
    const keybinding2 = new KeybindingImpl({
      commandId: 'test.command2',
      combo: { key: 'G', ctrl: true }
    })

    store.addUserKeybinding(keybinding1)
    store.addUserKeybinding(keybinding2)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(keybinding2.combo)).toEqual(keybinding2)
  })

  it('should not throw an error when unsetting non-existent keybindings', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'H', alt: true, shift: true }
    })

    expect(() => store.unsetKeybinding(keybinding)).not.toThrow()
  })

  it('should not throw an error when unsetting unknown keybinding', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'I', ctrl: true }
    })
    store.addUserKeybinding(keybinding)

    expect(() =>
      store.unsetKeybinding(
        new KeybindingImpl({
          commandId: 'test.foo',
          combo: { key: 'I', ctrl: true }
        })
      )
    ).not.toThrow()
  })

  it('should remove unset keybinding when adding back a default keybinding', () => {
    const store = useKeybindingStore()
    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'I', ctrl: true }
    })

    store.addDefaultKeybinding(defaultKeybinding)
    expect(store.keybindings).toHaveLength(1)

    store.unsetKeybinding(defaultKeybinding)
    expect(store.keybindings).toHaveLength(0)

    store.addUserKeybinding(defaultKeybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(defaultKeybinding.combo)).toEqual(
      defaultKeybinding
    )
  })

  it('Should accept same keybinding from default and user', () => {
    const store = useKeybindingStore()
    const keybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'J', ctrl: true }
    })
    store.addDefaultKeybinding(keybinding)
    store.addUserKeybinding(keybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(keybinding.combo)).toEqual(keybinding)
  })

  it('Should keep previously customized keybindings after default keybindings change', () => {
    const store = useKeybindingStore()

    const userUnsetKeybindings = [
      new KeybindingImpl({
        commandId: 'foo',
        combo: { key: 'K', ctrl: true }
      })
    ]

    const userNewKeybindings = [
      new KeybindingImpl({
        commandId: 'foo',
        combo: { key: 'A', ctrl: true }
      })
    ]

    const newCoreKeybindings = [
      new KeybindingImpl({
        commandId: 'foo',
        combo: { key: 'A', ctrl: true }
      })
    ]

    for (const keybinding of newCoreKeybindings) {
      store.addDefaultKeybinding(keybinding)
    }

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(userNewKeybindings[0].combo)).toEqual(
      userNewKeybindings[0]
    )

    for (const keybinding of userUnsetKeybindings) {
      store.unsetKeybinding(keybinding)
    }

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(userNewKeybindings[0].combo)).toEqual(
      userNewKeybindings[0]
    )

    for (const keybinding of userNewKeybindings) {
      store.addUserKeybinding(keybinding)
    }

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(userNewKeybindings[0].combo)).toEqual(
      userNewKeybindings[0]
    )
  })

  it('should replace the previous keybinding with a new one for the same combo and unset the old command', () => {
    const store = useKeybindingStore()

    const oldKeybinding = new KeybindingImpl({
      commandId: 'command1',
      combo: { key: 'A', ctrl: true }
    })

    store.addUserKeybinding(oldKeybinding)

    const newKeybinding = new KeybindingImpl({
      commandId: 'command2',
      combo: { key: 'A', ctrl: true }
    })

    store.updateKeybindingOnCommand(newKeybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybinding(newKeybinding.combo)?.commandId).toBe('command2')
    expect(store.getKeybindingsByCommandId('command1')).toHaveLength(0)
  })

  it('should return false when no default or current keybinding exists during reset', () => {
    const store = useKeybindingStore()
    const result = store.resetKeybindingForCommand('nonexistent.command')
    expect(result).toBe(false)
  })

  it('should return false when current keybinding equals default keybinding', () => {
    const store = useKeybindingStore()
    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'L', ctrl: true }
    })

    store.addDefaultKeybinding(defaultKeybinding)
    const result = store.resetKeybindingForCommand('test.command')

    expect(result).toBe(false)
    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybindingByCommandId('test.command')).toEqual(
      defaultKeybinding
    )
  })

  it('should unset user keybinding when no default keybinding exists and return true', () => {
    const store = useKeybindingStore()
    const userKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'M', ctrl: true }
    })

    store.addUserKeybinding(userKeybinding)
    expect(store.keybindings).toHaveLength(1)

    const result = store.resetKeybindingForCommand('test.command')

    expect(result).toBe(true)
    expect(store.keybindings).toHaveLength(0)
  })

  it('should restore default keybinding when user has overridden it and return true', () => {
    const store = useKeybindingStore()

    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'N', ctrl: true }
    })

    const userKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'O', alt: true }
    })

    store.addDefaultKeybinding(defaultKeybinding)
    store.updateKeybindingOnCommand(userKeybinding)

    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybindingByCommandId('test.command')).toEqual(
      userKeybinding
    )

    const result = store.resetKeybindingForCommand('test.command')

    expect(result).toBe(true)
    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybindingByCommandId('test.command')).toEqual(
      defaultKeybinding
    )
  })

  it('should remove unset record and restore default keybinding when user has unset it', () => {
    const store = useKeybindingStore()

    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'P', ctrl: true }
    })

    store.addDefaultKeybinding(defaultKeybinding)

    store.unsetKeybinding(defaultKeybinding)
    expect(store.keybindings).toHaveLength(0)

    const serializedCombo = defaultKeybinding.combo.serialize()
    const userUnsetKeybindings = store.getUserUnsetKeybindings()
    expect(userUnsetKeybindings[serializedCombo]).toBeTruthy()
    expect(
      userUnsetKeybindings[serializedCombo].equals(defaultKeybinding)
    ).toBe(true)

    const result = store.resetKeybindingForCommand('test.command')

    expect(result).toBe(true)
    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybindingByCommandId('test.command')).toEqual(
      defaultKeybinding
    )

    expect(store.getUserUnsetKeybindings()[serializedCombo]).toBeUndefined()
  })

  it('should handle complex scenario with both unset and user keybindings', () => {
    const store = useKeybindingStore()

    const defaultKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'Q', ctrl: true }
    })
    store.addDefaultKeybinding(defaultKeybinding)

    store.unsetKeybinding(defaultKeybinding)
    expect(store.keybindings).toHaveLength(0)

    const userKeybinding = new KeybindingImpl({
      commandId: 'test.command',
      combo: { key: 'R', alt: true }
    })
    store.addUserKeybinding(userKeybinding)
    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybindingByCommandId('test.command')).toEqual(
      userKeybinding
    )

    const result = store.resetKeybindingForCommand('test.command')

    expect(result).toBe(true)
    expect(store.keybindings).toHaveLength(1)
    expect(store.getKeybindingByCommandId('test.command')).toEqual(
      defaultKeybinding
    )
  })
})
