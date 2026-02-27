<template>
  <TreeExplorerTreeNode :node="node">
    <template #actions>
      <Button
        variant="muted-textonly"
        size="icon-sm"
        :aria-label="$t('icon.bookmark')"
        @click.stop="handleBookmarkClick"
      >
        <i
          :class="[
            isBookmarked ? 'pi pi-bookmark-fill' : 'pi pi-bookmark',
            'size-3.5'
          ]"
        />
      </Button>
    </template>
  </TreeExplorerTreeNode>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import TreeExplorerTreeNode from '@/components/common/TreeExplorerTreeNode.vue'
import Button from '@/components/ui/button/Button.vue'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowBookmarkStore } from '@/platform/workflow/management/stores/workflowStore'
import type { RenderedTreeExplorerNode } from '@/types/treeExplorerTypes'

const { node } = defineProps<{
  node: RenderedTreeExplorerNode<ComfyWorkflow>
}>()

const workflowBookmarkStore = useWorkflowBookmarkStore()
const isBookmarked = computed(
  () => node.data && workflowBookmarkStore.isBookmarked(node.data.path)
)

const handleBookmarkClick = async () => {
  if (node.data) {
    await workflowBookmarkStore.toggleBookmarked(node.data.path)
  }
}
</script>
