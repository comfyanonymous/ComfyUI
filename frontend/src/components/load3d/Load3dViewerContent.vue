<template>
  <div
    ref="viewerContentRef"
    class="flex w-full"
    :class="[maximized ? 'h-full' : 'h-[70vh]']"
    @mouseenter="viewer.handleMouseEnter"
    @mouseleave="viewer.handleMouseLeave"
  >
    <div class="relative flex-1">
      <div
        ref="containerRef"
        class="absolute h-full w-full"
        @resize="viewer.handleResize"
        @dragover.prevent.stop="handleDragOver"
        @dragleave.stop="handleDragLeave"
        @drop.prevent.stop="handleDrop"
      />
      <AnimationControls
        v-if="viewer.animations.value && viewer.animations.value.length > 0"
        v-model:animations="viewer.animations.value"
        v-model:playing="viewer.playing.value"
        v-model:selected-speed="viewer.selectedSpeed.value"
        v-model:selected-animation="viewer.selectedAnimation.value"
        v-model:animation-progress="viewer.animationProgress.value"
        v-model:animation-duration="viewer.animationDuration.value"
        @seek="viewer.handleSeek"
      />
      <div
        v-if="isDragging"
        class="pointer-events-none absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      >
        <div
          class="rounded-lg border-2 border-dashed border-blue-400 bg-blue-500/20 px-6 py-4 text-lg font-medium text-blue-100"
        >
          {{ dragMessage }}
        </div>
      </div>
    </div>

    <div class="flex w-72 flex-col">
      <div class="flex-1 overflow-y-auto p-4">
        <div class="space-y-2">
          <div class="space-y-4 p-2">
            <SceneControls
              v-model:background-color="viewer.backgroundColor.value"
              v-model:show-grid="viewer.showGrid.value"
              v-model:background-render-mode="viewer.backgroundRenderMode.value"
              v-model:fov="viewer.fov.value"
              :has-background-image="viewer.hasBackgroundImage.value"
              :disable-background-upload="viewer.isStandaloneMode.value"
              @update-background-image="viewer.handleBackgroundImageUpdate"
            />
          </div>

          <div class="space-y-4 p-2">
            <ModelControls
              v-model:up-direction="viewer.upDirection.value"
              v-model:material-mode="viewer.materialMode.value"
              :hide-material-mode="viewer.isSplatModel.value"
              :is-ply-model="viewer.isPlyModel.value"
            />
          </div>

          <div class="space-y-4 p-2">
            <CameraControls
              v-model:camera-type="viewer.cameraType.value"
              v-model:fov="viewer.fov.value"
            />
          </div>

          <div v-if="!viewer.isSplatModel.value" class="space-y-4 p-2">
            <LightControls
              v-model:light-intensity="viewer.lightIntensity.value"
            />
          </div>

          <div v-if="!viewer.isSplatModel.value" class="space-y-4 p-2">
            <ExportControls @export-model="viewer.exportModel" />
          </div>
        </div>
      </div>

      <div class="p-4">
        <div class="flex gap-2">
          <Button variant="secondary" @click="handleCancel">
            <i class="pi pi-times" />
            {{ t('g.cancel') }}
          </Button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, toRaw } from 'vue'
import { useI18n } from 'vue-i18n'

import AnimationControls from '@/components/load3d/controls/AnimationControls.vue'
import CameraControls from '@/components/load3d/controls/viewer/ViewerCameraControls.vue'
import ExportControls from '@/components/load3d/controls/viewer/ViewerExportControls.vue'
import LightControls from '@/components/load3d/controls/viewer/ViewerLightControls.vue'
import ModelControls from '@/components/load3d/controls/viewer/ViewerModelControls.vue'
import SceneControls from '@/components/load3d/controls/viewer/ViewerSceneControls.vue'
import Button from '@/components/ui/button/Button.vue'
import { useLoad3dDrag } from '@/composables/useLoad3dDrag'
import { useLoad3dViewer } from '@/composables/useLoad3dViewer'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useLoad3dService } from '@/services/load3dService'
import { useDialogStore } from '@/stores/dialogStore'

const { t } = useI18n()
const props = defineProps<{
  node?: LGraphNode
  modelUrl?: string
}>()

const viewerContentRef = ref<HTMLDivElement>()
const containerRef = ref<HTMLDivElement>()
const maximized = ref(false)
const mutationObserver = ref<MutationObserver | null>(null)

const isStandaloneMode = !props.node && props.modelUrl

// Use sync version since useLoad3dViewer is already imported (module is loaded)
const viewer = props.node
  ? useLoad3dService().getOrCreateViewerSync(toRaw(props.node), useLoad3dViewer)
  : useLoad3dViewer()

const { isDragging, dragMessage, handleDragOver, handleDragLeave, handleDrop } =
  useLoad3dDrag({
    onModelDrop: async (file) => {
      await viewer.handleModelDrop(file)
    },
    disabled: viewer.isPreview.value || !!isStandaloneMode
  })

onMounted(async () => {
  if (!containerRef.value) return

  if (isStandaloneMode && props.modelUrl) {
    await viewer.initializeStandaloneViewer(containerRef.value, props.modelUrl)
  } else if (props.node) {
    const source = useLoad3dService().getLoad3d(props.node)
    if (source) {
      await viewer.initializeViewer(containerRef.value, source)
    }
  }

  if (viewerContentRef.value) {
    mutationObserver.value = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.type === 'attributes' &&
          mutation.attributeName === 'maximized'
        ) {
          maximized.value =
            (mutation.target as HTMLElement).getAttribute('maximized') ===
            'true'

          setTimeout(() => {
            viewer.refreshViewport()
          }, 0)
        }
      })
    })

    mutationObserver.value.observe(viewerContentRef.value, {
      attributes: true,
      attributeFilter: ['maximized']
    })
  }

  window.addEventListener('resize', viewer.handleResize)
})

const handleCancel = () => {
  if (!isStandaloneMode) {
    viewer.restoreInitialState()
  }
  useDialogStore().closeDialog()
}

onBeforeUnmount(() => {
  window.removeEventListener('resize', viewer.handleResize)

  if (mutationObserver.value) {
    mutationObserver.value.disconnect()
    mutationObserver.value = null
  }

  // we will manually cleanup the viewer in dialog close handler
})
</script>

<style scoped>
:deep(.p-panel-content) {
  padding: 0;
}
</style>
