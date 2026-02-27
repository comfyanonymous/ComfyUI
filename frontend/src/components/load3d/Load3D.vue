<template>
  <div
    class="relative size-full"
    @mouseenter="handleMouseEnter"
    @mouseleave="handleMouseLeave"
    @pointerdown.stop
    @pointermove.stop
    @pointerup.stop
  >
    <Load3DScene
      v-if="node"
      :initialize-load3d="initializeLoad3d"
      :cleanup="cleanup"
      :loading="loading"
      :loading-message="loadingMessage"
      :on-model-drop="isPreview ? undefined : handleModelDrop"
      :is-preview="isPreview"
    />
    <div class="pointer-events-none absolute top-0 left-0 size-full">
      <Load3DControls
        v-model:scene-config="sceneConfig"
        v-model:model-config="modelConfig"
        v-model:camera-config="cameraConfig"
        v-model:light-config="lightConfig"
        :is-splat-model="isSplatModel"
        :is-ply-model="isPlyModel"
        :has-skeleton="hasSkeleton"
        @update-background-image="handleBackgroundImageUpdate"
        @export-model="handleExportModel"
      />
      <AnimationControls
        v-if="animations && animations.length > 0"
        v-model:animations="animations"
        v-model:playing="playing"
        v-model:selected-speed="selectedSpeed"
        v-model:selected-animation="selectedAnimation"
        v-model:animation-progress="animationProgress"
        v-model:animation-duration="animationDuration"
        @seek="handleSeek"
      />
    </div>
    <div
      v-if="enable3DViewer && node"
      class="pointer-events-auto absolute top-12 right-2 z-20"
    >
      <ViewerControls :node="node as LGraphNode" />
    </div>

    <div
      v-if="!isPreview"
      class="pointer-events-auto absolute right-2 z-20"
      :class="{
        'top-12': !enable3DViewer,
        'top-24': enable3DViewer
      }"
    >
      <RecordingControls
        v-model:is-recording="isRecording"
        v-model:has-recording="hasRecording"
        v-model:recording-duration="recordingDuration"
        @start-recording="handleStartRecording"
        @stop-recording="handleStopRecording"
        @export-recording="handleExportRecording"
        @clear-recording="handleClearRecording"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import type { Ref } from 'vue'

import Load3DControls from '@/components/load3d/Load3DControls.vue'
import Load3DScene from '@/components/load3d/Load3DScene.vue'
import AnimationControls from '@/components/load3d/controls/AnimationControls.vue'
import RecordingControls from '@/components/load3d/controls/RecordingControls.vue'
import ViewerControls from '@/components/load3d/controls/ViewerControls.vue'
import { useLoad3d } from '@/composables/useLoad3d'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import { app } from '@/scripts/app'
import type { ComponentWidget } from '@/scripts/domWidget'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

const props = defineProps<{
  widget: ComponentWidget<string[]> | SimplifiedWidget
  nodeId?: NodeId
}>()

function isComponentWidget(
  widget: ComponentWidget<string[]> | SimplifiedWidget
): widget is ComponentWidget<string[]> {
  return 'node' in widget && widget.node !== undefined
}

const node = ref<LGraphNode | null>(null)

if (isComponentWidget(props.widget)) {
  node.value = props.widget.node
} else if (props.nodeId) {
  onMounted(() => {
    node.value = app.rootGraph?.getNodeById(props.nodeId!) || null
  })
}

const {
  // configs
  sceneConfig,
  modelConfig,
  cameraConfig,
  lightConfig,

  // other state
  isRecording,
  isPreview,
  isSplatModel,
  isPlyModel,
  hasSkeleton,
  hasRecording,
  recordingDuration,
  animations,
  playing,
  selectedSpeed,
  selectedAnimation,
  animationProgress,
  animationDuration,
  loading,
  loadingMessage,

  // Methods
  initializeLoad3d,
  handleMouseEnter,
  handleMouseLeave,
  handleStartRecording,
  handleStopRecording,
  handleExportRecording,
  handleClearRecording,
  handleSeek,
  handleBackgroundImageUpdate,
  handleExportModel,
  handleModelDrop,
  cleanup
} = useLoad3d(node as Ref<LGraphNode | null>)

const enable3DViewer = computed(() =>
  useSettingStore().get('Comfy.Load3D.3DViewerEnable')
)
</script>
