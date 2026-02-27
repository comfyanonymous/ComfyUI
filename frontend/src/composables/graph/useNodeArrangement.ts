import { useI18n } from 'vue-i18n'

import type { Direction } from '@/lib/litegraph/src/interfaces'
import { alignNodes, distributeNodes } from '@/lib/litegraph/src/utils/arrange'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { isLGraphNode } from '@/utils/litegraphUtil'

import { useCanvasRefresh } from './useCanvasRefresh'

interface AlignOption {
  name: string
  localizedName: string
  value: Direction
  icon: string
}

interface DistributeOption {
  name: string
  localizedName: string
  value: boolean // true for horizontal, false for vertical
  icon: string
}

/**
 * Composable for handling node alignment and distribution
 */
export function useNodeArrangement() {
  const { t } = useI18n()
  const canvasStore = useCanvasStore()
  const canvasRefresh = useCanvasRefresh()
  const alignOptions: AlignOption[] = [
    {
      name: 'top',
      localizedName: t('contextMenu.Top'),
      value: 'top',
      icon: 'icon-[lucide--align-start-vertical]'
    },
    {
      name: 'bottom',
      localizedName: t('contextMenu.Bottom'),
      value: 'bottom',
      icon: 'icon-[lucide--align-end-vertical]'
    },
    {
      name: 'left',
      localizedName: t('contextMenu.Left'),
      value: 'left',
      icon: 'icon-[lucide--align-start-horizontal]'
    },
    {
      name: 'right',
      localizedName: t('contextMenu.Right'),
      value: 'right',
      icon: 'icon-[lucide--align-end-horizontal]'
    }
  ]

  const distributeOptions: DistributeOption[] = [
    {
      name: 'horizontal',
      localizedName: t('contextMenu.Horizontal'),
      value: true,
      icon: 'icon-[lucide--align-center-horizontal]'
    },
    {
      name: 'vertical',
      localizedName: t('contextMenu.Vertical'),
      value: false,
      icon: 'icon-[lucide--align-center-vertical]'
    }
  ]

  const applyAlign = (alignOption: AlignOption) => {
    const selectedNodes = Array.from(canvasStore.selectedItems).filter((item) =>
      isLGraphNode(item)
    )

    if (selectedNodes.length === 0) {
      return
    }

    const newPositions = alignNodes(selectedNodes, alignOption.value)
    canvasStore.canvas?.repositionNodesVueMode(newPositions)

    canvasRefresh.refreshCanvas()
  }

  const applyDistribute = (distributeOption: DistributeOption) => {
    const selectedNodes = Array.from(canvasStore.selectedItems).filter((item) =>
      isLGraphNode(item)
    )

    if (selectedNodes.length < 2) {
      return
    }

    const newPositions = distributeNodes(selectedNodes, distributeOption.value)
    canvasStore.canvas?.repositionNodesVueMode(newPositions)
    canvasRefresh.refreshCanvas()
  }

  return {
    alignOptions,
    distributeOptions,
    applyAlign,
    applyDistribute
  }
}
