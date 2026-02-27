<template>
  <div v-if="isOnlyHasNodes" class="p-4">
    <NodeSettings :nodes="theNodes" />
  </div>
  <div v-else class="border-t border-interface-stroke">
    <PropertiesAccordionItem
      v-if="hasNodes"
      class="border-b border-interface-stroke"
      :label="$t('rightSidePanel.nodes')"
    >
      <NodeSettings :nodes="theNodes" class="p-4" />
    </PropertiesAccordionItem>
    <PropertiesAccordionItem
      v-if="hasGroups"
      class="border-b border-interface-stroke"
      :label="$t('rightSidePanel.groups')"
    >
      <NodeSettings :nodes="theGroups" class="p-4" />
    </PropertiesAccordionItem>
  </div>
</template>

<script setup lang="ts">
/**
 * If we only need to show settings for Nodes,
 * there's no need to wrap them in PropertiesAccordionItem,
 * making the UI cleaner.
 * But if there are multiple types of settings,
 * it's better to wrap them; otherwise,
 * the UI would look messy.
 */
import { computed } from 'vue'
import type { Raw } from 'vue'

import type { LGraphGroup } from '@/lib/litegraph/src/LGraphGroup'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { Positionable } from '@/lib/litegraph/src/interfaces'
import { isLGraphGroup, isLGraphNode } from '@/utils/litegraphUtil'

import PropertiesAccordionItem from '../layout/PropertiesAccordionItem.vue'
import NodeSettings from './NodeSettings.vue'

const props = defineProps<{
  nodes: Raw<Positionable>[]
}>()

const theNodes = computed<LGraphNode[]>(() =>
  props.nodes.filter((node) => isLGraphNode(node))
)

const theGroups = computed<LGraphGroup[]>(() =>
  props.nodes.filter((node) => isLGraphGroup(node))
)

const hasGroups = computed(() => theGroups.value.length > 0)
const hasNodes = computed(() => theNodes.value.length > 0)
const isOnlyHasNodes = computed(() => hasNodes.value && !hasGroups.value)
</script>
