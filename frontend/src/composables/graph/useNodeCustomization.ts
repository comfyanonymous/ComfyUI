import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import {
  LGraphCanvas,
  LGraphNode,
  LiteGraph,
  RenderShape,
  isColorable
} from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { adjustColor } from '@/utils/colorUtil'

import { useCanvasRefresh } from './useCanvasRefresh'

interface ColorOption {
  name: string
  localizedName: string
  value: {
    dark: string
    light: string
  }
}

interface ShapeOption {
  name: string
  localizedName: string
  value: RenderShape
}

/**
 * Composable for handling node color and shape customization
 */
export function useNodeCustomization() {
  const { t } = useI18n()
  const canvasStore = useCanvasStore()
  const colorPaletteStore = useColorPaletteStore()
  const canvasRefresh = useCanvasRefresh()
  const isLightTheme = computed(
    () => colorPaletteStore.completedActivePalette.light_theme
  )

  const toLightThemeColor = (color: string) =>
    adjustColor(color, { lightness: 0.5 })

  // Color options
  const NO_COLOR_OPTION: ColorOption = {
    name: 'noColor',
    localizedName: t('color.noColor'),
    value: {
      dark: LiteGraph.NODE_DEFAULT_BGCOLOR,
      light: toLightThemeColor(LiteGraph.NODE_DEFAULT_BGCOLOR)
    }
  }

  const colorOptions: ColorOption[] = [
    NO_COLOR_OPTION,
    ...Object.entries(LGraphCanvas.node_colors).map(([name, color]) => ({
      name,
      localizedName: t(`color.${name}`),
      value: {
        dark: color.bgcolor,
        light: toLightThemeColor(color.bgcolor)
      }
    }))
  ]

  // Shape options
  const shapeOptions: ShapeOption[] = [
    {
      name: 'default',
      localizedName: t('shape.default'),
      value: RenderShape.ROUND
    },
    {
      name: 'box',
      localizedName: t('shape.box'),
      value: RenderShape.BOX
    },
    {
      name: 'card',
      localizedName: t('shape.CARD'),
      value: RenderShape.CARD
    }
  ]

  const applyColor = (colorOption: ColorOption | null) => {
    const colorName = colorOption?.name ?? NO_COLOR_OPTION.name
    const canvasColorOption =
      colorName === NO_COLOR_OPTION.name
        ? null
        : LGraphCanvas.node_colors[colorName]

    for (const item of canvasStore.selectedItems) {
      if (isColorable(item)) {
        item.setColorOption(canvasColorOption)
      }
    }

    canvasRefresh.refreshCanvas()
  }

  const applyShape = (shapeOption: ShapeOption) => {
    const selectedNodes = Array.from(canvasStore.selectedItems).filter(
      (item): item is LGraphNode => item instanceof LGraphNode
    )

    if (selectedNodes.length === 0) {
      return
    }

    selectedNodes.forEach((node) => {
      node.shape = shapeOption.value
    })

    canvasRefresh.refreshCanvas()
  }

  const getCurrentColor = (): ColorOption | null => {
    const selectedItems = Array.from(canvasStore.selectedItems)
    if (selectedItems.length === 0) return null

    // Get color from first colorable item
    const firstColorableItem = selectedItems.find((item) => isColorable(item))
    if (!firstColorableItem || !isColorable(firstColorableItem)) return null

    // Get the current color option from the colorable item
    const currentColorOption = firstColorableItem.getColorOption()
    const currentBgColor = currentColorOption?.bgcolor ?? null

    // Find matching color option
    return (
      colorOptions.find(
        (option) =>
          option.value.dark === currentBgColor ||
          option.value.light === currentBgColor
      ) ?? NO_COLOR_OPTION
    )
  }

  const getCurrentShape = (): ShapeOption | null => {
    const selectedNodes = Array.from(canvasStore.selectedItems).filter(
      (item): item is LGraphNode => item instanceof LGraphNode
    )

    if (selectedNodes.length === 0) return null

    const firstNode = selectedNodes[0]
    const currentShape = firstNode.shape ?? RenderShape.ROUND

    return (
      shapeOptions.find((option) => option.value === currentShape) ??
      shapeOptions[0]
    )
  }

  return {
    colorOptions,
    shapeOptions,
    applyColor,
    applyShape,
    getCurrentColor,
    getCurrentShape,
    isLightTheme
  }
}
