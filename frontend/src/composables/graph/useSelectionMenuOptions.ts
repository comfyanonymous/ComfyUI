import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useCommandStore } from '@/stores/commandStore'

import { useFrameNodes } from './useFrameNodes'
import { BadgeVariant } from './useMoreOptionsMenu'
import type { MenuOption } from './useMoreOptionsMenu'
import { useNodeArrangement } from './useNodeArrangement'
import { useSelectionOperations } from './useSelectionOperations'
import { useSubgraphOperations } from './useSubgraphOperations'

/**
 * Composable for selection-related menu operations
 */
export function useSelectionMenuOptions() {
  const { t } = useI18n()
  const {
    copySelection,
    duplicateSelection,
    deleteSelection,
    renameSelection
  } = useSelectionOperations()

  const { alignOptions, distributeOptions, applyAlign, applyDistribute } =
    useNodeArrangement()

  const { convertToSubgraph, unpackSubgraph, addSubgraphToLibrary } =
    useSubgraphOperations()

  const { frameNodes } = useFrameNodes()

  const alignSubmenu = computed(() =>
    alignOptions.map((align) => ({
      label: align.localizedName,
      icon: align.icon,
      action: () => applyAlign(align)
    }))
  )

  const distributeSubmenu = computed(() =>
    distributeOptions.map((distribute) => ({
      label: distribute.localizedName,
      icon: distribute.icon,
      action: () => applyDistribute(distribute)
    }))
  )

  const getBasicSelectionOptions = (): MenuOption[] => [
    {
      label: t('contextMenu.Rename'),
      action: renameSelection
    },
    {
      label: t('contextMenu.Copy'),
      shortcut: 'Ctrl+C',
      action: copySelection
    },
    {
      label: t('contextMenu.Duplicate'),
      shortcut: 'Ctrl+D',
      action: duplicateSelection
    }
  ]

  const getSubgraphOptions = ({
    hasSubgraphs,
    hasMultipleSelection
  }: {
    hasSubgraphs: boolean
    hasMultipleSelection: boolean
  }): MenuOption[] => {
    const convertOption: MenuOption = {
      label: t('contextMenu.Convert to Subgraph'),
      icon: 'icon-[lucide--shrink]',
      action: convertToSubgraph,
      badge: BadgeVariant.NEW
    }

    const options: MenuOption[] = []
    const showConvertOption = !hasSubgraphs || hasMultipleSelection

    if (showConvertOption) {
      options.push(convertOption)
    }

    if (hasSubgraphs) {
      if (!hasMultipleSelection) {
        options.push({
          label: t('contextMenu.Add Subgraph to Library'),
          icon: 'icon-[lucide--folder-plus]',
          action: addSubgraphToLibrary
        })
      }
      options.push({
        label: t('contextMenu.Unpack Subgraph'),
        icon: 'icon-[lucide--expand]',
        action: unpackSubgraph
      })
    }

    return options
  }

  const getMultipleNodesOptions = (): MenuOption[] => {
    const convertToGroupNodes = () => {
      const commandStore = useCommandStore()
      void commandStore.execute(
        'Comfy.GroupNode.ConvertSelectedNodesToGroupNode'
      )
    }

    return [
      {
        label: t('contextMenu.Convert to Group Node'),
        icon: 'icon-[lucide--group]',
        action: convertToGroupNodes,
        badge: BadgeVariant.DEPRECATED
      },
      {
        label: t('g.frameNodes'),
        icon: 'icon-[lucide--frame]',
        action: frameNodes
      }
    ]
  }

  const getAlignmentOptions = (): MenuOption[] => [
    {
      label: t('contextMenu.Align Selected To'),
      icon: 'icon-[lucide--align-start-horizontal]',
      hasSubmenu: true,
      submenu: alignSubmenu.value,
      action: () => {}
    },
    {
      label: t('contextMenu.Distribute Nodes'),
      icon: 'icon-[lucide--align-center-horizontal]',
      hasSubmenu: true,
      submenu: distributeSubmenu.value,
      action: () => {}
    }
  ]

  const getDeleteOption = (): MenuOption => ({
    label: t('contextMenu.Delete'),
    icon: 'icon-[lucide--trash-2]',
    shortcut: 'Delete',
    action: deleteSelection
  })

  return {
    getBasicSelectionOptions,
    getSubgraphOptions,
    getMultipleNodesOptions,
    getDeleteOption,
    getAlignmentOptions,
    alignSubmenu,
    distributeSubmenu
  }
}
