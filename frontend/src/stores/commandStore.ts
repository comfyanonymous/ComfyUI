import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { useErrorHandling } from '@/composables/useErrorHandling'
import type { KeybindingImpl } from '@/platform/keybindings/keybinding'
import { useKeybindingStore } from '@/platform/keybindings/keybindingStore'
import type { ComfyExtension } from '@/types/comfy'

export interface ComfyCommand {
  id: string
  function: (metadata?: Record<string, unknown>) => void | Promise<void>

  label?: string | (() => string)
  icon?: string | (() => string)
  tooltip?: string | (() => string)
  menubarLabel?: string | (() => string) // Menubar item label, if different from command label
  versionAdded?: string
  confirmation?: string // If non-nullish, this command will prompt for confirmation
  source?: string
  active?: () => boolean // Getter to check if the command is active/toggled on
  category?: 'essentials' | 'view-controls' // For shortcuts panel organization
}

export class ComfyCommandImpl implements ComfyCommand {
  id: string
  function: (metadata?: Record<string, unknown>) => void | Promise<void>
  _label?: string | (() => string)
  _icon?: string | (() => string)
  _tooltip?: string | (() => string)
  _menubarLabel?: string | (() => string)
  versionAdded?: string
  confirmation?: string
  source?: string
  active?: () => boolean
  category?: 'essentials' | 'view-controls'

  constructor(command: ComfyCommand) {
    this.id = command.id
    this.function = command.function
    this._label = command.label
    this._icon = command.icon
    this._tooltip = command.tooltip
    this._menubarLabel = command.menubarLabel ?? command.label
    this.versionAdded = command.versionAdded
    this.confirmation = command.confirmation
    this.source = command.source
    this.active = command.active
    this.category = command.category
  }

  get label() {
    return typeof this._label === 'function' ? this._label() : this._label
  }

  get icon() {
    return typeof this._icon === 'function' ? this._icon() : this._icon
  }

  get tooltip() {
    return typeof this._tooltip === 'function' ? this._tooltip() : this._tooltip
  }

  get menubarLabel() {
    return typeof this._menubarLabel === 'function'
      ? this._menubarLabel()
      : this._menubarLabel
  }

  get keybinding(): KeybindingImpl | null {
    return useKeybindingStore().getKeybindingByCommandId(this.id)
  }
}

export const useCommandStore = defineStore('command', () => {
  const commandsById = ref<Record<string, ComfyCommandImpl>>({})
  const commands = computed(() => Object.values(commandsById.value))

  const registerCommand = (command: ComfyCommand) => {
    if (commandsById.value[command.id]) {
      console.warn(`Command ${command.id} already registered`)
    }
    commandsById.value[command.id] = new ComfyCommandImpl(command)
  }

  const registerCommands = (commands: ComfyCommand[]) => {
    for (const command of commands) {
      registerCommand(command)
    }
  }

  const getCommand = (command: string) => {
    return commandsById.value[command]
  }

  const { wrapWithErrorHandlingAsync } = useErrorHandling()
  const execute = async (
    commandId: string,
    options?: {
      errorHandler?: (error: unknown) => void
      metadata?: Record<string, unknown>
    }
  ) => {
    const command = getCommand(commandId)
    if (command) {
      await wrapWithErrorHandlingAsync(
        () => command.function(options?.metadata),
        options?.errorHandler
      )()
    } else {
      throw new Error(`Command ${commandId} not found`)
    }
  }

  const isRegistered = (command: string) => {
    return !!commandsById.value[command]
  }

  const loadExtensionCommands = (extension: ComfyExtension) => {
    if (extension.commands) {
      for (const command of extension.commands) {
        registerCommand({
          ...command,
          source: extension.name
        })
      }
    }
  }

  const formatKeySequence = (command: ComfyCommandImpl): string => {
    const sequences = command.keybinding?.combo.getKeySequences() || []
    return sequences
      .map((seq) => seq.replace(/Control/g, 'Ctrl').replace(/Shift/g, 'Shift'))
      .join(' + ')
  }

  return {
    commands,
    execute,
    getCommand,
    registerCommand,
    registerCommands,
    isRegistered,
    loadExtensionCommands,
    formatKeySequence
  }
})
