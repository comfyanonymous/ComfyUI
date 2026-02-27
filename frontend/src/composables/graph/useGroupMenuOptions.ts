import { useI18n } from 'vue-i18n'

import { LGraphEventMode } from '@/lib/litegraph/src/litegraph'
import type { LGraphGroup, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

import { useCanvasRefresh } from './useCanvasRefresh'
import type { MenuOption } from './useMoreOptionsMenu'
import { useNodeCustomization } from './useNodeCustomization'

/**
 * Composable for group-related menu operations
 */
export function useGroupMenuOptions() {
  const { t } = useI18n()
  const canvasStore = useCanvasStore()
  const workflowStore = useWorkflowStore()
  const settingStore = useSettingStore()
  const canvasRefresh = useCanvasRefresh()
  const { shapeOptions, colorOptions, isLightTheme } = useNodeCustomization()

  const getFitGroupToNodesOption = (groupContext: LGraphGroup): MenuOption => ({
    label: 'Fit Group To Nodes',
    icon: 'icon-[lucide--move-diagonal-2]',
    action: () => {
      try {
        groupContext.recomputeInsideNodes()
      } catch (e) {
        console.warn('Failed to recompute group nodes:', e)
        return
      }

      const padding = settingStore.get('Comfy.GroupSelectedNodes.Padding')
      groupContext.resizeTo(groupContext.children, padding)
      groupContext.graph?.change()
      canvasStore.canvas?.setDirty(true, true)
      workflowStore.activeWorkflow?.changeTracker?.checkState()
    }
  })

  const getGroupShapeOptions = (
    groupContext: LGraphGroup,
    bump: () => void
  ): MenuOption => ({
    label: t('contextMenu.Shape'),
    icon: 'icon-[lucide--box]',
    hasSubmenu: true,
    submenu: shapeOptions.map((shape) => ({
      label: shape.localizedName,
      action: () => {
        const nodes = (groupContext.nodes || []) as LGraphNode[]
        nodes.forEach((node) => (node.shape = shape.value))
        canvasRefresh.refreshCanvas()
        bump()
      }
    }))
  })

  const getGroupColorOptions = (
    groupContext: LGraphGroup,
    bump: () => void
  ): MenuOption => ({
    label: t('contextMenu.Color'),
    icon: 'icon-[lucide--palette]',
    hasSubmenu: true,
    submenu: colorOptions.map((colorOption) => ({
      label: colorOption.localizedName,
      color: isLightTheme.value
        ? colorOption.value.light
        : colorOption.value.dark,
      action: () => {
        groupContext.color = isLightTheme.value
          ? colorOption.value.light
          : colorOption.value.dark
        canvasRefresh.refreshCanvas()
        bump()
      }
    }))
  })

  const getGroupModeOptions = (
    groupContext: LGraphGroup,
    bump: () => void
  ): MenuOption[] => {
    const options: MenuOption[] = []

    try {
      groupContext.recomputeInsideNodes()
    } catch (e) {
      console.warn('Failed to recompute group nodes for mode options:', e)
      return options
    }

    const groupNodes = (groupContext.nodes || []) as LGraphNode[]
    if (!groupNodes.length) return options

    // Check if all nodes have the same mode
    let allSame = true
    for (let i = 1; i < groupNodes.length; i++) {
      if (groupNodes[i].mode !== groupNodes[0].mode) {
        allSame = false
        break
      }
    }

    const createModeAction = (label: string, mode: LGraphEventMode) => ({
      label: t(`selectionToolbox.${label}`),
      icon:
        mode === LGraphEventMode.BYPASS
          ? 'icon-[lucide--ban]'
          : mode === LGraphEventMode.NEVER
            ? 'icon-[lucide--zap-off]'
            : 'icon-[lucide--play]',
      action: () => {
        groupNodes.forEach((n) => {
          n.mode = mode
        })
        canvasStore.canvas?.setDirty(true, true)
        groupContext.graph?.change()
        workflowStore.activeWorkflow?.changeTracker?.checkState()
        bump()
      }
    })

    if (allSame) {
      const current = groupNodes[0].mode
      switch (current) {
        case LGraphEventMode.ALWAYS:
          options.push(
            createModeAction('Set Group Nodes to Never', LGraphEventMode.NEVER)
          )
          options.push(
            createModeAction('Bypass Group Nodes', LGraphEventMode.BYPASS)
          )
          break
        case LGraphEventMode.NEVER:
          options.push(
            createModeAction(
              'Set Group Nodes to Always',
              LGraphEventMode.ALWAYS
            )
          )
          options.push(
            createModeAction('Bypass Group Nodes', LGraphEventMode.BYPASS)
          )
          break
        case LGraphEventMode.BYPASS:
          options.push(
            createModeAction(
              'Set Group Nodes to Always',
              LGraphEventMode.ALWAYS
            )
          )
          options.push(
            createModeAction('Set Group Nodes to Never', LGraphEventMode.NEVER)
          )
          break
        default:
          options.push(
            createModeAction(
              'Set Group Nodes to Always',
              LGraphEventMode.ALWAYS
            )
          )
          options.push(
            createModeAction('Set Group Nodes to Never', LGraphEventMode.NEVER)
          )
          options.push(
            createModeAction('Bypass Group Nodes', LGraphEventMode.BYPASS)
          )
          break
      }
    } else {
      options.push(
        createModeAction('Set Group Nodes to Always', LGraphEventMode.ALWAYS)
      )
      options.push(
        createModeAction('Set Group Nodes to Never', LGraphEventMode.NEVER)
      )
      options.push(
        createModeAction('Bypass Group Nodes', LGraphEventMode.BYPASS)
      )
    }

    return options
  }

  return {
    getFitGroupToNodesOption,
    getGroupShapeOptions,
    getGroupColorOptions,
    getGroupModeOptions
  }
}
