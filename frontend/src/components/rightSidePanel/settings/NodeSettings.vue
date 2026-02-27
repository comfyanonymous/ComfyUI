<template>
  <div class="space-y-4 text-sm text-muted-foreground">
    <SetNodeState
      v-if="isNodes(targetNodes)"
      :nodes="targetNodes"
      @changed="handleChanged"
    />
    <SetNodeColor :nodes="targetNodes" @changed="handleChanged" />
    <SetPinned :nodes="targetNodes" @changed="handleChanged" />
  </div>
</template>

<script setup lang="ts">
import { shallowRef, watchEffect } from 'vue'

import type { LGraphGroup, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { isLGraphGroup } from '@/utils/litegraphUtil'

import SetNodeColor from './SetNodeColor.vue'
import SetNodeState from './SetNodeState.vue'
import SetPinned from './SetPinned.vue'

const props = defineProps<{
  /**
   * - If the item is a Group, Node State cannot be set
   * as Groups do not have a 'mode' property.
   *
   * - The nodes array can contain either all Nodes or all Groups,
   * but it must not be a mix of both.
   */
  nodes?: LGraphNode[] | LGraphGroup[]
}>()

const targetNodes = shallowRef<LGraphNode[] | LGraphGroup[]>([])
watchEffect(() => {
  if (props.nodes) {
    targetNodes.value = props.nodes
  } else {
    targetNodes.value = []
  }
})

const canvasStore = useCanvasStore()

function isNodes(nodes: LGraphNode[] | LGraphGroup[]): nodes is LGraphNode[] {
  return !nodes.some((node) => isLGraphGroup(node))
}

function handleChanged() {
  /**
   * This is not a random comment—it's crucial.
   * Otherwise, the UI cannot update correctly.
   * There is a bug with triggerRef here, so we can't use triggerRef.
   * We'll work around it for now and later submit a Vue issue and pull request to fix it.
   */
  targetNodes.value = targetNodes.value.slice()

  canvasStore.canvas?.setDirty(true, true)
}
</script>
