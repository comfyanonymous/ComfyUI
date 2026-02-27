<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'

import FieldSwitch from './FieldSwitch.vue'

type PickedNode = Pick<LGraphNode, 'pinned' | 'pin'>

/**
 * Good design limits dependencies and simplifies the interface of the abstraction layer.
 * Here, we only care about the pinned and pin methods,
 * and do not concern ourselves with other methods.
 */
const { nodes } = defineProps<{ nodes: PickedNode[] }>()
const emit = defineEmits<{ (e: 'changed'): void }>()

const { t } = useI18n()

// Pinned state
const isPinned = computed<boolean>({
  get() {
    return nodes.some((node) => node.pinned)
  },
  set(value) {
    nodes.forEach((node) => node.pin(value))
    emit('changed')
  }
})
</script>

<template>
  <FieldSwitch v-model="isPinned" :label="t('rightSidePanel.pinned')" />
</template>
