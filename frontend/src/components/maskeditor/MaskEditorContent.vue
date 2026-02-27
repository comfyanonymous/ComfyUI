<template>
  <div
    ref="containerRef"
    class="maskEditor-dialog-root flex h-full w-full flex-col"
    @contextmenu.prevent
    @dragstart="handleDragStart"
    @keydown.stop
  >
    <div
      id="maskEditorCanvasContainer"
      ref="canvasContainerRef"
      @contextmenu.prevent
    >
      <canvas
        ref="imgCanvasRef"
        class="absolute top-0 left-0 w-full h-full z-0"
        @contextmenu.prevent
      />
      <canvas
        ref="rgbCanvasRef"
        class="absolute top-0 left-0 w-full h-full z-10"
        @contextmenu.prevent
      />
      <canvas
        ref="maskCanvasRef"
        class="absolute top-0 left-0 w-full h-full z-30"
        @contextmenu.prevent
      />
      <!-- GPU Preview Canvas -->
      <canvas
        ref="gpuCanvasRef"
        class="absolute top-0 left-0 w-full h-full pointer-events-none"
        :class="{
          'z-20': store.activeLayer === 'rgb',
          'z-40': store.activeLayer === 'mask'
        }"
      />
      <div ref="canvasBackgroundRef" class="bg-white w-full h-full" />
    </div>

    <div class="maskEditor-ui-container flex min-h-0 flex-1 flex-col">
      <div class="flex min-h-0 flex-1 overflow-hidden">
        <ToolPanel
          v-if="initialized"
          ref="toolPanelRef"
          :tool-manager="toolManager!"
        />

        <PointerZone
          v-if="initialized"
          :tool-manager="toolManager!"
          :pan-zoom="panZoom!"
        />

        <SidePanel
          v-if="initialized"
          ref="sidePanelRef"
          :tool-manager="toolManager!"
        />
      </div>
    </div>

    <BrushCursor v-if="initialized" :container-ref="containerRef" />
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

import { useImageLoader } from '@/composables/maskeditor/useImageLoader'
import { useKeyboard } from '@/composables/maskeditor/useKeyboard'
import { useMaskEditorLoader } from '@/composables/maskeditor/useMaskEditorLoader'
import { usePanAndZoom } from '@/composables/maskeditor/usePanAndZoom'
import { useToolManager } from '@/composables/maskeditor/useToolManager'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useDialogStore } from '@/stores/dialogStore'
import { useMaskEditorDataStore } from '@/stores/maskEditorDataStore'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

import BrushCursor from './BrushCursor.vue'
import PointerZone from './PointerZone.vue'
import SidePanel from './SidePanel.vue'
import ToolPanel from './ToolPanel.vue'

const { node } = defineProps<{
  node: LGraphNode
}>()

const store = useMaskEditorStore()
const dataStore = useMaskEditorDataStore()
const dialogStore = useDialogStore()

const loader = useMaskEditorLoader()

const containerRef = ref<HTMLElement>()
const canvasContainerRef = ref<HTMLDivElement>()
const imgCanvasRef = ref<HTMLCanvasElement>()
const maskCanvasRef = ref<HTMLCanvasElement>()
const rgbCanvasRef = ref<HTMLCanvasElement>()
const gpuCanvasRef = ref<HTMLCanvasElement>()
const canvasBackgroundRef = ref<HTMLDivElement>()

const toolPanelRef = ref<InstanceType<typeof ToolPanel>>()
const sidePanelRef = ref<InstanceType<typeof SidePanel>>()

const initialized = ref(false)

const keyboard = useKeyboard()
const panZoom = usePanAndZoom()

const toolManager = useToolManager(keyboard, panZoom)

let resizeObserver: ResizeObserver | null = null

const handleDragStart = (event: DragEvent) => {
  if (event.ctrlKey) {
    event.preventDefault()
  }
}

const initUI = async () => {
  if (!containerRef.value) {
    console.error(
      '[MaskEditorContent] Cannot initialize - missing required refs'
    )
    return
  }

  if (
    !imgCanvasRef.value ||
    !maskCanvasRef.value ||
    !rgbCanvasRef.value ||
    !canvasContainerRef.value ||
    !canvasBackgroundRef.value
  ) {
    console.error('[MaskEditorContent] Cannot initialize - missing canvas refs')
    return
  }

  store.maskCanvas = maskCanvasRef.value
  store.rgbCanvas = rgbCanvasRef.value
  store.imgCanvas = imgCanvasRef.value
  store.canvasContainer = canvasContainerRef.value
  store.canvasBackground = canvasBackgroundRef.value

  try {
    await loader.loadFromNode(node)

    const imageLoader = useImageLoader()
    const image = await imageLoader.loadImages()

    await panZoom.initializeCanvasPanZoom(
      image,
      containerRef.value,
      toolPanelRef.value?.$el as HTMLElement | undefined,
      sidePanelRef.value?.$el as HTMLElement | undefined
    )

    store.canvasHistory.saveInitialState()

    // Initialize GPU resources
    if (toolManager.brushDrawing) {
      await toolManager.brushDrawing.initGPUResources()
      if (gpuCanvasRef.value && toolManager.brushDrawing.initPreviewCanvas) {
        // Match preview canvas resolution to mask canvas
        gpuCanvasRef.value.width = maskCanvasRef.value.width
        gpuCanvasRef.value.height = maskCanvasRef.value.height

        toolManager.brushDrawing.initPreviewCanvas(gpuCanvasRef.value)
      }
    }

    initialized.value = true
  } catch (error) {
    console.error('[MaskEditorContent] Initialization failed:', error)
    dialogStore.closeDialog()
  }
}

onMounted(() => {
  keyboard.addListeners()

  if (containerRef.value) {
    resizeObserver = new ResizeObserver(async () => {
      if (panZoom) {
        await panZoom.invalidatePanZoom()
      }
    })
    resizeObserver.observe(containerRef.value)
  }

  void initUI()
})

onBeforeUnmount(() => {
  toolManager.brushDrawing.saveBrushSettings()

  keyboard?.removeListeners()

  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }

  store.canvasHistory.clearStates()

  store.resetState()
  dataStore.reset()
})
</script>

<style scoped>
.maskEditor-dialog-root {
  position: relative;
  overflow: hidden;
}

.maskEditor-ui-container {
  position: relative;
  z-index: 1;
}

:deep(#maskEditorCanvasContainer) {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}
</style>
