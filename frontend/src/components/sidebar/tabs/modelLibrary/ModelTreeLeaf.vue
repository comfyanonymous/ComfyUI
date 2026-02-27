<template>
  <div ref="container" class="model-lib-node-container h-full w-full">
    <TreeExplorerTreeNode :node="node">
      <template #before-label>
        <span v-if="modelPreviewUrl" class="model-lib-model-icon-container">
          <span
            class="model-lib-model-icon"
            :style="{ backgroundImage: `url(${modelPreviewUrl})` }"
          />
        </span>
      </template>
    </TreeExplorerTreeNode>

    <teleport v-if="showPreview" to="#model-library-model-preview-container">
      <div class="model-lib-model-preview" :style="modelPreviewStyle">
        <ModelPreview ref="previewRef" :model-def="modelDef" />
      </div>
    </teleport>
  </div>
</template>

<script setup lang="ts">
import type { CSSProperties } from 'vue'
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'

import TreeExplorerTreeNode from '@/components/common/TreeExplorerTreeNode.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { ComfyModelDef } from '@/stores/modelStore'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'

import ModelPreview from './ModelPreview.vue'

const props = defineProps<{
  node: RenderedTreeExplorerNode<ComfyModelDef>
}>()

// Note: The leaf node should always have a model definition on node.data.
const modelDef = computed<ComfyModelDef>(() => props.node.data!)

const modelPreviewUrl = computed(() => {
  if (modelDef.value.image) {
    return modelDef.value.image
  }
  const folder = modelDef.value.directory
  const path_index = modelDef.value.path_index
  const extension = modelDef.value.file_name.split('.').pop()
  const filename = modelDef.value.file_name.replace(`.${extension}`, '.webp')
  const encodedFilename = encodeURIComponent(filename).replace(/%2F/g, '/')
  return `/api/experiment/models/preview/${folder}/${path_index}/${encodedFilename}`
})

const previewRef = ref<InstanceType<typeof ModelPreview> | null>(null)
const modelPreviewStyle = ref<CSSProperties>({
  position: 'absolute',
  top: '0px',
  left: '0px'
})

const settingStore = useSettingStore()
const sidebarLocation = computed<'left' | 'right'>(() =>
  settingStore.get('Comfy.Sidebar.Location')
)

const handleModelHover = async () => {
  const hoverTarget = modelContentElement.value
  if (!hoverTarget) return

  const targetRect = hoverTarget.getBoundingClientRect()

  const previewHeight = previewRef.value?.$el.offsetHeight || 0
  const availableSpaceBelow = window.innerHeight - targetRect.bottom

  modelPreviewStyle.value.top =
    previewHeight > availableSpaceBelow
      ? `${Math.max(0, targetRect.top - (previewHeight - availableSpaceBelow) - 20)}px`
      : `${targetRect.top - 40}px`
  if (sidebarLocation.value === 'left') {
    modelPreviewStyle.value.left = `${targetRect.right}px`
  } else {
    modelPreviewStyle.value.left = `${targetRect.left - 400}px`
  }

  await modelDef.value.load()
}

const container = ref<HTMLElement | undefined>()
const modelContentElement = ref<HTMLElement | undefined>()
const isHovered = ref(false)

const showPreview = computed(() => {
  return (
    isHovered.value &&
    modelDef.value &&
    modelDef.value.has_loaded_metadata &&
    (modelDef.value.author ||
      modelDef.value.simplified_file_name != modelDef.value.title ||
      modelDef.value.description ||
      modelDef.value.usage_hint ||
      modelDef.value.trigger_phrase ||
      modelDef.value.image)
  )
})

const handleMouseEnter = async () => {
  isHovered.value = true
  await nextTick()
  await handleModelHover()
}
const handleMouseLeave = () => {
  isHovered.value = false
}
onMounted(async () => {
  modelContentElement.value =
    container.value?.closest('.p-tree-node-content') ?? undefined
  modelContentElement.value?.addEventListener('mouseenter', handleMouseEnter)
  modelContentElement.value?.addEventListener('mouseleave', handleMouseLeave)
  await modelDef.value.load()
})

onUnmounted(() => {
  modelContentElement.value?.removeEventListener('mouseenter', handleMouseEnter)
  modelContentElement.value?.removeEventListener('mouseleave', handleMouseLeave)
})
</script>

<style scoped>
.model-lib-model-icon-container {
  display: inline-block;
  position: relative;
  left: 0;
  height: 1.5rem;
  vertical-align: top;
  width: 0;
}
.model-lib-model-icon {
  background-size: cover;
  background-position: center;
  display: inline-block;
  position: relative;
  left: -2.2rem;
  top: -0.1rem;
  height: 1.7rem;
  width: 1.7rem;
  vertical-align: top;
}
</style>
