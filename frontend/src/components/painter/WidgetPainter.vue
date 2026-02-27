<template>
  <div
    class="widget-expands flex h-full w-full flex-col gap-1"
    @pointerdown.stop
    @pointermove.stop
    @pointerup.stop
  >
    <div
      class="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-lg bg-node-component-surface"
    >
      <div class="relative max-h-full w-full" :style="canvasContainerStyle">
        <img
          v-if="inputImageUrl"
          :src="inputImageUrl"
          class="absolute inset-0 size-full"
          draggable="false"
          @load="handleInputImageLoad"
          @dragstart.prevent
        />
        <canvas
          ref="canvasEl"
          class="absolute inset-0 size-full cursor-none touch-none"
          @pointerdown="handlePointerDown"
          @pointermove="handlePointerMove"
          @pointerup="handlePointerUp"
          @pointerenter="handlePointerEnter"
          @pointerleave="handlePointerLeave"
        />
        <div
          v-show="cursorVisible"
          class="pointer-events-none absolute left-0 top-0 rounded-full border border-black/60 shadow-[0_0_0_1px_rgba(255,255,255,0.8)]"
          :style="cursorStyle"
        />
      </div>
    </div>

    <div
      v-if="isImageInputConnected"
      class="text-center text-xs text-muted-foreground"
    >
      {{ canvasWidth }} x {{ canvasHeight }}
    </div>

    <div
      ref="controlsEl"
      :class="
        cn(
          'grid shrink-0 gap-x-1 gap-y-1',
          compact ? 'grid-cols-1' : 'grid-cols-[auto_1fr]'
        )
      "
    >
      <div
        v-if="!compact"
        class="flex w-28 items-center truncate text-sm text-muted-foreground"
      >
        {{ $t('painter.tool') }}
      </div>
      <div
        class="flex h-8 items-center gap-1 rounded-sm bg-component-node-widget-background p-1"
      >
        <Button
          variant="textonly"
          size="unset"
          :class="
            cn(
              'flex-1 self-stretch px-2 text-xs transition-colors',
              tool === PAINTER_TOOLS.BRUSH
                ? 'rounded-sm bg-component-node-widget-background-selected text-base-foreground'
                : 'text-node-text-muted hover:text-node-text'
            )
          "
          @click="tool = PAINTER_TOOLS.BRUSH"
        >
          {{ $t('painter.brush') }}
        </Button>
        <Button
          variant="textonly"
          size="unset"
          :class="
            cn(
              'flex-1 self-stretch px-2 text-xs transition-colors',
              tool === PAINTER_TOOLS.ERASER
                ? 'rounded-sm bg-component-node-widget-background-selected text-base-foreground'
                : 'text-node-text-muted hover:text-node-text'
            )
          "
          @click="tool = PAINTER_TOOLS.ERASER"
        >
          {{ $t('painter.eraser') }}
        </Button>
      </div>

      <div
        v-if="!compact"
        class="flex w-28 items-center truncate text-sm text-muted-foreground"
      >
        {{ $t('painter.size') }}
      </div>
      <div
        class="flex h-8 items-center gap-2 rounded-lg bg-component-node-widget-background pr-2 pl-3"
      >
        <Slider
          :model-value="[brushSize]"
          :min="1"
          :max="200"
          :step="1"
          class="flex-1"
          @update:model-value="(v) => v?.length && (brushSize = v[0])"
        />
        <span class="w-8 text-center text-xs text-node-text-muted">{{
          brushSize
        }}</span>
      </div>

      <template v-if="tool === PAINTER_TOOLS.BRUSH">
        <div
          class="flex w-28 items-center truncate text-sm text-muted-foreground"
        >
          {{ $t('painter.color') }}
        </div>
        <div
          class="flex h-8 w-full items-center gap-2 rounded-lg bg-component-node-widget-background px-4"
        >
          <input
            type="color"
            :value="brushColorDisplay"
            class="h-4 w-8 cursor-pointer appearance-none overflow-hidden rounded-full border-none bg-transparent [&::-webkit-color-swatch-wrapper]:p-0 [&::-webkit-color-swatch]:border-none [&::-webkit-color-swatch]:rounded-full [&::-moz-color-swatch]:border-none [&::-moz-color-swatch]:rounded-full"
            @input="
              (e) => (brushColorDisplay = (e.target as HTMLInputElement).value)
            "
          />
          <span class="min-w-[4ch] truncate text-xs">{{
            brushColorDisplay
          }}</span>
          <span class="ml-auto flex items-center text-xs text-node-text-muted">
            <input
              type="number"
              :value="brushOpacityPercent"
              min="0"
              max="100"
              step="1"
              class="w-7 appearance-none border-0 bg-transparent text-right text-xs text-node-text-muted outline-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none [-moz-appearance:textfield]"
              @click.prevent
              @change="
                (e) => {
                  const val = Math.min(
                    100,
                    Math.max(0, Number((e.target as HTMLInputElement).value))
                  )
                  brushOpacityPercent = val
                  ;(e.target as HTMLInputElement).value = String(val)
                }
              "
            />%</span
          >
        </div>

        <div
          class="flex w-28 items-center truncate text-sm text-muted-foreground"
        >
          {{ $t('painter.hardness') }}
        </div>
        <div
          class="flex h-8 items-center gap-2 rounded-lg bg-component-node-widget-background pr-2 pl-3"
        >
          <Slider
            :model-value="[brushHardnessPercent]"
            :min="0"
            :max="100"
            :step="1"
            class="flex-1"
            @update:model-value="
              (v) => v?.length && (brushHardnessPercent = v[0])
            "
          />
          <span class="w-8 text-center text-xs text-node-text-muted"
            >{{ brushHardnessPercent }}%</span
          >
        </div>
      </template>

      <template v-if="!isImageInputConnected">
        <div
          class="flex w-28 items-center truncate text-sm text-muted-foreground"
        >
          {{ $t('painter.width') }}
        </div>
        <div
          class="flex h-8 items-center gap-2 rounded-lg bg-component-node-widget-background pr-2 pl-3"
        >
          <Slider
            :model-value="[canvasWidth]"
            :min="64"
            :max="4096"
            :step="64"
            class="flex-1"
            @update:model-value="(v) => v?.length && (canvasWidth = v[0])"
          />
          <span class="w-10 text-center text-xs text-node-text-muted">{{
            canvasWidth
          }}</span>
        </div>

        <div
          class="flex w-28 items-center truncate text-sm text-muted-foreground"
        >
          {{ $t('painter.height') }}
        </div>
        <div
          class="flex h-8 items-center gap-2 rounded-lg bg-component-node-widget-background pr-2 pl-3"
        >
          <Slider
            :model-value="[canvasHeight]"
            :min="64"
            :max="4096"
            :step="64"
            class="flex-1"
            @update:model-value="(v) => v?.length && (canvasHeight = v[0])"
          />
          <span class="w-10 text-center text-xs text-node-text-muted">{{
            canvasHeight
          }}</span>
        </div>

        <div
          class="flex w-28 items-center truncate text-sm text-muted-foreground"
        >
          {{ $t('painter.background') }}
        </div>
        <div
          class="flex h-8 w-full items-center gap-2 rounded-lg bg-component-node-widget-background px-4"
        >
          <input
            type="color"
            :value="backgroundColorDisplay"
            class="h-4 w-8 cursor-pointer appearance-none overflow-hidden rounded-full border-none bg-transparent [&::-webkit-color-swatch-wrapper]:p-0 [&::-webkit-color-swatch]:border-none [&::-webkit-color-swatch]:rounded-full [&::-moz-color-swatch]:border-none [&::-moz-color-swatch]:rounded-full"
            @input="
              (e) =>
                (backgroundColorDisplay = (e.target as HTMLInputElement).value)
            "
          />
          <span class="min-w-[4ch] truncate text-xs">{{
            backgroundColorDisplay
          }}</span>
        </div>
      </template>

      <Button
        variant="secondary"
        size="md"
        :class="
          cn(
            'gap-2 rounded-lg border border-component-node-border bg-component-node-background text-xs text-muted-foreground hover:text-base-foreground',
            !compact && 'col-span-2'
          )
        "
        @click="handleClear"
      >
        <i class="icon-[lucide--undo-2]" />
        {{ $t('painter.clear') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useElementSize } from '@vueuse/core'
import { computed, useTemplateRef } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import Slider from '@/components/ui/slider/Slider.vue'
import { PAINTER_TOOLS, usePainter } from '@/composables/painter/usePainter'
import { toHexFromFormat } from '@/utils/colorUtil'
import { cn } from '@/utils/tailwindUtil'

const { nodeId } = defineProps<{
  nodeId: string
}>()

const modelValue = defineModel<string>({ default: '' })

const canvasEl = useTemplateRef<HTMLCanvasElement>('canvasEl')
const controlsEl = useTemplateRef<HTMLDivElement>('controlsEl')
const { width: controlsWidth } = useElementSize(controlsEl)
const compact = computed(
  () => controlsWidth.value > 0 && controlsWidth.value < 350
)

const {
  tool,
  brushSize,
  brushColor,
  brushOpacity,
  brushHardness,
  backgroundColor,
  canvasWidth,
  canvasHeight,
  cursorX,
  cursorY,
  cursorVisible,
  displayBrushSize,
  inputImageUrl,
  isImageInputConnected,
  handlePointerDown,
  handlePointerMove,
  handlePointerUp,
  handlePointerEnter,
  handlePointerLeave,
  handleInputImageLoad,
  handleClear
} = usePainter(nodeId, { canvasEl, modelValue })

const canvasContainerStyle = computed(() => ({
  aspectRatio: `${canvasWidth.value} / ${canvasHeight.value}`,
  backgroundColor: isImageInputConnected.value
    ? undefined
    : backgroundColor.value
}))

const cursorStyle = computed(() => {
  const size = displayBrushSize.value
  const x = cursorX.value - size / 2
  const y = cursorY.value - size / 2
  return {
    width: `${size}px`,
    height: `${size}px`,
    transform: `translate(${x}px, ${y}px)`
  }
})

const brushOpacityPercent = computed({
  get: () => Math.round(brushOpacity.value * 100),
  set: (val: number) => {
    brushOpacity.value = val / 100
  }
})

const brushHardnessPercent = computed({
  get: () => Math.round(brushHardness.value * 100),
  set: (val: number) => {
    brushHardness.value = val / 100
  }
})

const brushColorDisplay = computed({
  get: () => toHexFromFormat(brushColor.value, 'hex'),
  set: (val: unknown) => {
    brushColor.value = toHexFromFormat(val, 'hex')
  }
})

const backgroundColorDisplay = computed({
  get: () => toHexFromFormat(backgroundColor.value, 'hex'),
  set: (val: unknown) => {
    backgroundColor.value = toHexFromFormat(val, 'hex')
  }
})
</script>
