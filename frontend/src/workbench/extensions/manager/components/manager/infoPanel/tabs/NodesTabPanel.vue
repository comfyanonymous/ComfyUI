<template>
  <div class="flex flex-col gap-1 text-sm">
    <template v-if="mappedNodeDefs?.length">
      <div class="flex flex-col gap-2">
        <div
          v-for="nodeDef in mappedNodeDefs"
          :key="createNodeDefKey(nodeDef)"
          class="scale-75"
        >
          <NodePreview :node-def="nodeDef" class="min-w-full!" />
        </div>
      </div>
    </template>
    <template v-else-if="isLoading">
      <ProgressSpinner />
    </template>
    <template v-else-if="nodeNames.length">
      <div v-for="node in nodeNames" :key="node" class="truncate text-muted">
        {{ node }}
      </div>
    </template>
    <template v-else>
      <NoResultsPlaceholder
        :title="$t('manager.noNodesFound')"
        :message="$t('manager.noNodesFoundDescription')"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
import { whenever } from '@vueuse/core'
import ProgressSpinner from 'primevue/progressspinner'
import { computed, ref, shallowRef, useId } from 'vue'

import NoResultsPlaceholder from '@/components/common/NoResultsPlaceholder.vue'
import NodePreview from '@/components/node/NodePreview.vue'
import { useComfyRegistryStore } from '@/stores/comfyRegistryStore'
import type { components, operations } from '@/types/comfyRegistryTypes'
import { registryToFrontendV2NodeDef } from '@/utils/mapperUtil'

type ListComfyNodesResponse =
  operations['ListComfyNodes']['responses'][200]['content']['application/json']['comfy_nodes']

const { nodePack, nodeNames } = defineProps<{
  nodePack: components['schemas']['Node']
  nodeNames: string[]
}>()

const { getNodeDefs } = useComfyRegistryStore()

const isLoading = ref(false)
const registryNodeDefs = shallowRef<ListComfyNodesResponse | null>(null)

const fetchNodeDefs = async () => {
  getNodeDefs.cancel()
  isLoading.value = true

  const { id: packId } = nodePack
  const version = nodePack.latest_version?.version

  if (!packId || !version) {
    registryNodeDefs.value = null
  } else {
    const response = await getNodeDefs.call({
      packId,
      version,
      page: 1,
      limit: 256
    })
    registryNodeDefs.value = response?.comfy_nodes ?? null
  }

  isLoading.value = false
}

whenever(() => nodePack, fetchNodeDefs, { immediate: true, deep: true })

const toFrontendNodeDef = (nodeDef: components['schemas']['ComfyNode']) => {
  try {
    return registryToFrontendV2NodeDef(nodeDef, nodePack)
  } catch (error) {
    return null
  }
}
const mappedNodeDefs = computed(() => {
  if (!registryNodeDefs.value) return null
  return registryNodeDefs.value
    .map(toFrontendNodeDef)
    .filter((nodeDef) => nodeDef !== null)
})

const createNodeDefKey = (nodeDef: components['schemas']['ComfyNode']) =>
  `${nodeDef.category}${nodeDef.comfy_node_name ?? useId()}`
</script>
