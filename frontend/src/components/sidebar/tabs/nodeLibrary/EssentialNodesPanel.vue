<template>
  <TabsContent value="essentials" class="flex-1 overflow-y-auto px-3 h-full">
    <div class="flex flex-col gap-2 pb-6">
      <CollapsibleRoot
        v-for="folder in folders"
        :key="folder.key"
        class="rounded-lg"
        :open="expandedKeys.includes(folder.key)"
        @update:open="toggleFolder(folder.key, $event)"
      >
        <CollapsibleTrigger
          class="group flex w-full cursor-pointer items-center justify-between border-0 bg-transparent py-3 px-1 text-xs font-medium tracking-wide text-muted-foreground h-8 box-content"
        >
          <span class="uppercase">{{ folder.label }}</span>
          <i
            :class="
              cn(
                'icon-[lucide--chevron-up] size-4 transition-transform duration-200',
                !expandedKeys.includes(folder.key) && '-rotate-180'
              )
            "
          />
        </CollapsibleTrigger>
        <CollapsibleContent
          class="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down"
        >
          <div
            class="grid grid-cols-[repeat(auto-fill,minmax(5rem,1fr))] gap-3"
          >
            <EssentialNodeCard
              v-for="node in folder.children"
              :key="node.key"
              :node="node"
              @click="emit('nodeClick', $event)"
            />
          </div>
        </CollapsibleContent>
      </CollapsibleRoot>
    </div>
  </TabsContent>
</template>

<script setup lang="ts">
import {
  CollapsibleContent,
  CollapsibleRoot,
  CollapsibleTrigger,
  TabsContent
} from 'reka-ui'
import { computed, ref, watch } from 'vue'

import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'
import { cn } from '@/utils/tailwindUtil'

import EssentialNodeCard from './EssentialNodeCard.vue'

const props = defineProps<{
  root: RenderedTreeExplorerNode<ComfyNodeDefImpl>
}>()

const expandedKeys = defineModel<string[]>('expandedKeys', { required: true })

const emit = defineEmits<{
  nodeClick: [node: RenderedTreeExplorerNode<ComfyNodeDefImpl>]
}>()

function flattenLeaves(
  node: RenderedTreeExplorerNode<ComfyNodeDefImpl>
): RenderedTreeExplorerNode<ComfyNodeDefImpl>[] {
  if (node.type === 'node') return [node]
  return node.children?.flatMap(flattenLeaves) ?? []
}

const folders = computed(() => {
  const topFolders =
    (props.root.children?.filter(
      (child) => child.type === 'folder'
    ) as RenderedTreeExplorerNode<ComfyNodeDefImpl>[]) ?? []

  return topFolders.map((folder) => ({
    ...folder,
    children: flattenLeaves(folder)
  }))
})

function toggleFolder(key: string, open: boolean) {
  if (open) {
    expandedKeys.value = [...expandedKeys.value, key]
  } else {
    expandedKeys.value = expandedKeys.value.filter((k) => k !== key)
  }
}

const hasAutoExpanded = ref(false)

watch(
  folders,
  (value) => {
    if (!hasAutoExpanded.value && value.length > 0) {
      hasAutoExpanded.value = true
      if (expandedKeys.value.length === 0) {
        expandedKeys.value = value.map((folder) => folder.key)
      }
    }
  },
  { immediate: true }
)
</script>
