<script setup lang="ts">
import { ref, useTemplateRef, watch } from 'vue'

import Load3DControls from '@/components/load3d/Load3DControls.vue'
import AnimationControls from '@/components/load3d/controls/AnimationControls.vue'
import { useLoad3dViewer } from '@/composables/useLoad3dViewer'

const { modelUrl } = defineProps<{
  modelUrl: string
}>()

const containerRef = useTemplateRef('containerRef')

const viewer = ref(useLoad3dViewer())

watch([containerRef, () => modelUrl], async () => {
  if (!containerRef.value || !modelUrl) return

  await viewer.value.initializeStandaloneViewer(containerRef.value, modelUrl)
})

//TODO: refactor to add control buttons
</script>
<template>
  <div
    ref="containerRef"
    class="relative w-full h-full"
    @mouseenter="viewer.handleMouseEnter"
    @mouseleave="viewer.handleMouseLeave"
    @resize="viewer.handleResize"
  >
    <div class="pointer-events-none absolute top-0 left-0 size-full">
      <Load3DControls
        v-model:scene-config="viewer"
        v-model:model-config="viewer"
        v-model:camera-config="viewer"
        v-model:light-config="viewer"
        :is-splat-model="viewer.isSplatModel"
        :is-ply-model="viewer.isPlyModel"
        :has-skeleton="viewer.hasSkeleton"
        @update-background-image="viewer.handleBackgroundImageUpdate"
        @export-model="viewer.exportModel"
      />
      <AnimationControls
        v-if="viewer.animations && viewer.animations.length > 0"
        v-model:animations="viewer.animations"
        v-model:playing="viewer.playing"
        v-model:selected-speed="viewer.selectedSpeed"
        v-model:selected-animation="viewer.selectedAnimation"
        v-model:animation-progress="viewer.animationProgress"
        v-model:animation-duration="viewer.animationDuration"
        @seek="viewer.handleSeek"
      />
    </div>
  </div>
</template>
