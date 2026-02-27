<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, ref, shallowRef } from 'vue'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import FormSearchInput from '@/renderer/extensions/vueNodes/widgets/components/form/FormSearchInput.vue'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'

import { computedSectionDataList, searchWidgetsAndNodes } from '../shared'
import type { NodeWidgetsListList } from '../shared'
import SectionWidgets from './SectionWidgets.vue'

const canvasStore = useCanvasStore()
const workflowStore = useWorkflowStore()

const nodes = computed((): LGraphNode[] => {
  // Depend on activeWorkflow to trigger recomputation when workflow changes
  void workflowStore.activeWorkflow?.path
  return (canvasStore.canvas?.graph?.nodes ?? []) as LGraphNode[]
})

const rightSidePanelStore = useRightSidePanelStore()
const { searchQuery } = storeToRefs(rightSidePanelStore)

const { widgetsSectionDataList } = computedSectionDataList(nodes)

const searchedWidgetsSectionDataList = shallowRef<NodeWidgetsListList>(
  widgetsSectionDataList.value
)
const isSearching = ref(false)
async function searcher(query: string) {
  const list = widgetsSectionDataList.value
  const target = searchedWidgetsSectionDataList
  isSearching.value = query.trim() !== ''
  target.value = searchWidgetsAndNodes(list, query)
}
</script>

<template>
  <div class="px-4 pt-1 pb-4 flex gap-2 border-b border-interface-stroke">
    <FormSearchInput
      v-model="searchQuery"
      :searcher
      :update-key="widgetsSectionDataList"
    />
  </div>
  <TransitionGroup tag="div" name="list-scale" class="relative">
    <div
      v-if="isSearching && searchedWidgetsSectionDataList.length === 0"
      class="text-sm text-muted-foreground px-4 text-center pt-5 pb-15"
    >
      {{ $t('rightSidePanel.noneSearchDesc') }}
    </div>
    <SectionWidgets
      v-for="{ node, widgets } in searchedWidgetsSectionDataList"
      :key="node.id"
      :node
      :widgets
      :collapse="!isSearching"
      :tooltip="
        isSearching || widgets.length
          ? ''
          : $t('rightSidePanel.inputsNoneTooltip')
      "
      show-locate-button
      class="border-b border-interface-stroke"
    />
  </TransitionGroup>
</template>
