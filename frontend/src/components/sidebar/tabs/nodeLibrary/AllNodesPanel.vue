<template>
  <TabsContent value="all" class="flex-1 overflow-y-auto h-full">
    <!-- Favorites section -->
    <template v-if="hasFavorites">
      <h3
        class="px-4 py-2 text-xs font-medium uppercase tracking-wide text-muted-foreground mb-0"
      >
        {{ $t('sideToolbar.nodeLibraryTab.sections.favorites') }}
      </h3>
      <TreeExplorerV2
        v-model:expanded-keys="expandedKeys"
        :root="favoritesRoot"
        show-context-menu
        @node-click="(node) => emit('nodeClick', node)"
        @add-to-favorites="handleAddToFavorites"
      />
    </template>

    <!-- Node sections -->
    <div v-for="(section, index) in sections" :key="section.title ?? index">
      <h3
        v-if="section.title"
        class="px-4 py-2 text-xs font-medium tracking-wide text-muted-foreground mb-0"
      >
        {{ section.title }}
      </h3>
      <TreeExplorerV2
        v-model:expanded-keys="expandedKeys"
        :root="section.root"
        show-context-menu
        @node-click="(node) => emit('nodeClick', node)"
        @add-to-favorites="handleAddToFavorites"
      />
    </div>
  </TabsContent>
</template>

<script setup lang="ts">
import { TabsContent } from 'reka-ui'
import { computed } from 'vue'

import TreeExplorerV2 from '@/components/common/TreeExplorerV2.vue'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import type {
  NodeLibrarySection,
  RenderedTreeExplorerNode,
  TreeNode
} from '@/types/treeExplorerTypes'

const { fillNodeInfo } = defineProps<{
  sections: NodeLibrarySection[]
  fillNodeInfo: (node: TreeNode) => RenderedTreeExplorerNode<ComfyNodeDefImpl>
}>()

const expandedKeys = defineModel<string[]>('expandedKeys', { required: true })

const emit = defineEmits<{
  nodeClick: [node: RenderedTreeExplorerNode<ComfyNodeDefImpl>]
}>()

const nodeBookmarkStore = useNodeBookmarkStore()

const hasFavorites = computed(
  () => (nodeBookmarkStore.bookmarkedRoot.children?.length ?? 0) > 0
)

const favoritesRoot = computed(() =>
  fillNodeInfo(nodeBookmarkStore.bookmarkedRoot)
)

function handleAddToFavorites(
  node: RenderedTreeExplorerNode<ComfyNodeDefImpl>
) {
  if (node.data) {
    nodeBookmarkStore.toggleBookmark(node.data)
  }
}
</script>
