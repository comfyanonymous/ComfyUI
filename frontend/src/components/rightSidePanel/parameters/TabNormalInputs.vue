<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, ref, shallowRef } from 'vue'
import { useI18n } from 'vue-i18n'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import FormSearchInput from '@/renderer/extensions/vueNodes/widgets/components/form/FormSearchInput.vue'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'

import { computedSectionDataList, searchWidgetsAndNodes } from '../shared'
import type { NodeWidgetsListList } from '../shared'
import SectionWidgets from './SectionWidgets.vue'

const { nodes, mustShowNodeTitle } = defineProps<{
  mustShowNodeTitle?: boolean
  nodes: LGraphNode[]
}>()

const { t } = useI18n()

const rightSidePanelStore = useRightSidePanelStore()
const { searchQuery } = storeToRefs(rightSidePanelStore)

const { widgetsSectionDataList, includesAdvanced } = computedSectionDataList(
  () => nodes
)

const advancedWidgetsSectionDataList = computed((): NodeWidgetsListList => {
  if (includesAdvanced.value) {
    return []
  }
  return nodes
    .map((node) => {
      const { widgets = [] } = node
      const advancedWidgets = widgets
        .filter(
          (w) =>
            !(w.options?.canvasOnly || w.options?.hidden) && w.options?.advanced
        )
        .map((widget) => ({ node, widget }))
      return { widgets: advancedWidgets, node }
    })
    .filter(({ widgets }) => widgets.length > 0)
})

const isMultipleNodesSelected = computed(
  () => widgetsSectionDataList.value.length > 1
)

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

const label = computed(() => {
  const sections = widgetsSectionDataList.value
  return !mustShowNodeTitle && sections.length === 1
    ? sections[0].widgets.length !== 0
      ? t('rightSidePanel.inputs')
      : t('rightSidePanel.inputsNone')
    : undefined // SectionWidgets display node titles by default
})

const advancedLabel = computed(() => {
  return !mustShowNodeTitle && !isMultipleNodesSelected.value
    ? t('rightSidePanel.advancedInputs')
    : undefined // SectionWidgets display node titles by default
})
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
      v-if="searchedWidgetsSectionDataList.length === 0"
      class="text-sm text-muted-foreground px-4 py-10 text-center"
    >
      {{
        isSearching
          ? t('rightSidePanel.noneSearchDesc')
          : t('rightSidePanel.nodesNoneDesc')
      }}
    </div>
    <SectionWidgets
      v-for="{ widgets, node } in searchedWidgetsSectionDataList"
      :key="node.id"
      :node
      :label
      :widgets
      :collapse="isMultipleNodesSelected && !isSearching"
      :show-locate-button="isMultipleNodesSelected"
      :tooltip="
        isSearching || widgets.length
          ? ''
          : t('rightSidePanel.inputsNoneTooltip')
      "
      class="border-b border-interface-stroke"
    />
  </TransitionGroup>
  <template v-if="advancedWidgetsSectionDataList.length > 0 && !isSearching">
    <SectionWidgets
      v-for="{ widgets, node } in advancedWidgetsSectionDataList"
      :key="`advanced-${node.id}`"
      :collapse="true"
      :node
      :label="advancedLabel"
      :widgets
      :show-locate-button="isMultipleNodesSelected"
      class="border-b border-interface-stroke"
    />
  </template>
</template>
