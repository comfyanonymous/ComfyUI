<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LGraphCanvas, LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { ColorOption } from '@/lib/litegraph/src/litegraph'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { adjustColor } from '@/utils/colorUtil'
import { cn } from '@/utils/tailwindUtil'

import LayoutField from './LayoutField.vue'

/**
 * Good design limits dependencies and simplifies the interface of the abstraction layer.
 * Here, we only care about the getColorOption and setColorOption methods,
 * and do not concern ourselves with other methods.
 */
type PickedNode = Pick<LGraphNode, 'getColorOption' | 'setColorOption'>

const { nodes } = defineProps<{ nodes: PickedNode[] }>()
const emit = defineEmits<{ (e: 'changed'): void }>()

const { t } = useI18n()

const colorPaletteStore = useColorPaletteStore()

type NodeColorOption = {
  name: string
  localizedName: () => string
  value: {
    dark: string
    light: string
    ringDark: string
    ringLight: string
  }
}

const nodeColorEntries = Object.entries(LGraphCanvas.node_colors)

function getColorValue(color: string): NodeColorOption['value'] {
  return {
    dark: adjustColor(color, { lightness: 0.3 }),
    light: adjustColor(color, { lightness: 0.4 }),
    ringDark: adjustColor(color, { lightness: 0.5 }),
    ringLight: adjustColor(color, { lightness: 0.1 })
  }
}

const NO_COLOR_OPTION: NodeColorOption = {
  name: 'noColor',
  localizedName: () => t('color.noColor'),
  value: getColorValue(LiteGraph.NODE_DEFAULT_BGCOLOR)
}

const colorOptions: NodeColorOption[] = [
  NO_COLOR_OPTION,
  ...nodeColorEntries.map(([name, color]) => ({
    name,
    localizedName: () => t(`color.${name}`),
    value: getColorValue(color.bgcolor)
  }))
]

const isLightTheme = computed(
  () => colorPaletteStore.completedActivePalette.light_theme
)

const nodeColor = computed<NodeColorOption['name'] | null>({
  get() {
    if (nodes.length === 0) return null
    const theColorOptions = nodes.map((item) => item.getColorOption())

    let colorOption: ColorOption | null | false = theColorOptions[0]
    if (!theColorOptions.every((option) => option === colorOption)) {
      colorOption = false
    }

    if (colorOption === false) return null
    if (colorOption == null || (!colorOption.bgcolor && !colorOption.color))
      return NO_COLOR_OPTION.name
    return (
      nodeColorEntries.find(
        ([_, color]) =>
          color.bgcolor === colorOption.bgcolor &&
          color.color === colorOption.color
      )?.[0] ?? null
    )
  },
  set(colorName) {
    if (colorName === null) return

    const canvasColorOption =
      colorName === NO_COLOR_OPTION.name
        ? null
        : LGraphCanvas.node_colors[colorName]

    for (const item of nodes) {
      item.setColorOption(canvasColorOption)
    }

    emit('changed')
  }
})
</script>

<template>
  <LayoutField :label="t('rightSidePanel.color')">
    <div
      class="bg-secondary-background border-none rounded-lg p-1 grid grid-cols-5 gap-1 justify-items-center"
    >
      <button
        v-for="option of colorOptions"
        :key="option.name"
        :class="
          cn(
            'size-8 rounded-lg bg-transparent border-0 outline-0 ring-0 text-left flex justify-center items-center cursor-pointer',
            option.name === nodeColor
              ? 'bg-interface-menu-component-surface-selected'
              : 'hover:bg-interface-menu-component-surface-selected'
          )
        "
        @click="nodeColor = option.name"
      >
        <div
          v-tooltip.top="option.localizedName()"
          :class="cn('size-4 rounded-full ring-2 ring-gray-500/10')"
          :style="{
            backgroundColor: isLightTheme
              ? option.value.light
              : option.value.dark,
            '--tw-ring-color':
              option.name === nodeColor
                ? isLightTheme
                  ? option.value.ringLight
                  : option.value.ringDark
                : undefined
          }"
          :data-testid="option.name"
        />
      </button>
    </div>
  </LayoutField>
</template>
