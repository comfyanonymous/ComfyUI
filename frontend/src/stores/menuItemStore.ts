import { whenever } from '@vueuse/core'
import { defineStore } from 'pinia'
import type { MenuItem } from 'primevue/menuitem'
import { ref } from 'vue'

import { CORE_MENU_COMMANDS } from '@/constants/coreMenuCommands'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { ComfyExtension } from '@/types/comfy'

import { useCommandStore } from './commandStore'

export const useMenuItemStore = defineStore('menuItem', () => {
  const canvasStore = useCanvasStore()
  const commandStore = useCommandStore()
  const menuItems = ref<MenuItem[]>([])
  const menuItemHasActiveStateChildren = ref<Record<string, boolean>>({})
  const hasSeenLinear = ref(false)

  whenever(
    () => canvasStore.linearMode,
    () => (hasSeenLinear.value = true),
    { immediate: true, once: true }
  )

  const registerMenuGroup = (path: string[], items: MenuItem[]) => {
    let currentLevel = menuItems.value

    // Traverse the path, creating nodes if necessary
    for (let i = 0; i < path.length; i++) {
      const segment = path[i]
      let found = currentLevel.find((item) => item.label === segment)

      if (!found) {
        // Create a new node if it doesn't exist
        found = {
          label: segment,
          key: segment,
          items: []
        }
        currentLevel.push(found)
      }

      // Ensure the found item has an 'items' array
      if (!found.items) {
        found.items = []
      }

      // Move to the next level
      currentLevel = found.items
    }

    if (currentLevel.length > 0) {
      currentLevel.push({
        separator: true
      })
    }
    // Add the new items to the last level
    currentLevel.push(...items)

    // Store if any of the children have active state as we will hide the icon if they do
    const parentPath = path.join('.')
    if (!menuItemHasActiveStateChildren.value[parentPath]) {
      menuItemHasActiveStateChildren.value[parentPath] = items.some(
        (item) => item.comfyCommand?.active
      )
    }
  }

  const registerCommands = (path: string[], commandIds: string[]) => {
    const items = commandIds
      .map((commandId) => commandStore.getCommand(commandId))
      .map(
        (command) =>
          ({
            command: () => commandStore.execute(command.id),
            label: command.menubarLabel,
            icon: command.icon,
            tooltip: command.tooltip,
            comfyCommand: command,
            parentPath: path.join('.')
          }) as MenuItem
      )
    registerMenuGroup(path, items)
  }

  const loadExtensionMenuCommands = (extension: ComfyExtension) => {
    if (!extension.menuCommands) {
      return
    }

    const extensionCommandIds = new Set(
      extension.commands?.map((command) => command.id) ?? []
    )
    extension.menuCommands.forEach((menuCommand) => {
      const commands = menuCommand.commands.filter((command) =>
        extensionCommandIds.has(command)
      )
      if (commands.length) {
        registerCommands(menuCommand.path, commands)
      }
    })
  }

  const registerCoreMenuCommands = () => {
    for (const [path, commands] of CORE_MENU_COMMANDS) {
      registerCommands(path, commands)
    }
  }

  return {
    menuItems,
    registerMenuGroup,
    registerCommands,
    loadExtensionMenuCommands,
    registerCoreMenuCommands,
    menuItemHasActiveStateChildren,
    hasSeenLinear
  }
})
