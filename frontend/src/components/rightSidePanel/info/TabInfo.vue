<script setup lang="ts">
import { computed } from 'vue'

import NodeHelpContent from '@/components/node/NodeHelpContent.vue'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useNodeDefStore } from '@/stores/nodeDefStore'

const { nodes } = defineProps<{
  nodes: LGraphNode[]
}>()
const node = computed(() => nodes[0])

const nodeDefStore = useNodeDefStore()

const nodeInfo = computed(() => {
  return nodeDefStore.fromLGraphNode(node.value)
})
</script>

<template>
  <div v-if="nodeInfo" class="p-3">
    <NodeHelpContent :node="nodeInfo" />
  </div>
</template>
