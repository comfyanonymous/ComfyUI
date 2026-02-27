<template>
  <Message
    v-if="hasMissingCoreNodes"
    severity="info"
    icon="pi pi-info-circle"
    class="mx-2 my-2"
    :pt="{
      root: { class: 'flex-col' },
      text: { class: 'flex-1' }
    }"
  >
    <div class="flex flex-col gap-2">
      <div>
        {{
          currentComfyUIVersion
            ? $t('loadWorkflowWarning.outdatedVersion', {
                version: currentComfyUIVersion
              })
            : $t('loadWorkflowWarning.outdatedVersionGeneric')
        }}
      </div>
      <div
        v-for="[version, nodes] in sortedMissingCoreNodes"
        :key="version"
        class="ml-4"
      >
        <div class="text-sm font-medium text-surface-600">
          {{
            $t('loadWorkflowWarning.coreNodesFromVersion', {
              version: version || 'unknown'
            })
          }}
        </div>
        <div class="ml-4 text-sm text-surface-500">
          {{ getUniqueNodeNames(nodes).join(', ') }}
        </div>
      </div>
    </div>
  </Message>
</template>

<script setup lang="ts">
import Message from 'primevue/message'
import { compare } from 'semver'
import { computed } from 'vue'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useSystemStatsStore } from '@/stores/systemStatsStore'

const { missingCoreNodes } = defineProps<{
  missingCoreNodes: Record<string, LGraphNode[]>
}>()

const systemStatsStore = useSystemStatsStore()

const hasMissingCoreNodes = computed(() => {
  return Object.keys(missingCoreNodes).length > 0
})

// Use computed for reactive version tracking
const currentComfyUIVersion = computed<string | null>(() => {
  if (!hasMissingCoreNodes.value) return null
  return systemStatsStore.systemStats?.system?.comfyui_version ?? null
})

const sortedMissingCoreNodes = computed(() => {
  return Object.entries(missingCoreNodes).sort(([a], [b]) => {
    // Sort by version in descending order (newest first)
    return compare(b, a) // Reversed for descending order
  })
})

const getUniqueNodeNames = (nodes: LGraphNode[]): string[] => {
  return nodes
    .reduce<string[]>((acc, node) => {
      if (node.type && !acc.includes(node.type)) {
        acc.push(node.type)
      }
      return acc
    }, [])
    .sort()
}
</script>
