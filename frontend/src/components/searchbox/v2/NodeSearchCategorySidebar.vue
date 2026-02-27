<template>
  <div class="flex min-h-0 flex-col overflow-y-auto py-2.5">
    <!-- Preset categories -->
    <div class="flex flex-col px-1">
      <button
        v-for="preset in topCategories"
        :key="preset.id"
        type="button"
        :data-testid="`category-${preset.id}`"
        :aria-current="selectedCategory === preset.id || undefined"
        :class="categoryBtnClass(preset.id)"
        @click="selectCategory(preset.id)"
      >
        {{ preset.label }}
      </button>
    </div>

    <!-- Source categories -->
    <div class="my-2 flex flex-col border-y border-border-subtle px-1 py-2">
      <button
        v-for="preset in sourceCategories"
        :key="preset.id"
        type="button"
        :data-testid="`category-${preset.id}`"
        :aria-current="selectedCategory === preset.id || undefined"
        :class="categoryBtnClass(preset.id)"
        @click="selectCategory(preset.id)"
      >
        {{ preset.label }}
      </button>
    </div>

    <!-- Category tree -->
    <div class="flex flex-col px-1">
      <NodeSearchCategoryTreeNode
        v-for="category in categoryTree"
        :key="category.key"
        :node="category"
        :selected-category="selectedCategory"
        :selected-collapsed="selectedCollapsed"
        @select="selectCategory"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import NodeSearchCategoryTreeNode, {
  CATEGORY_SELECTED_CLASS,
  CATEGORY_UNSELECTED_CLASS
} from '@/components/searchbox/v2/NodeSearchCategoryTreeNode.vue'
import type { CategoryNode } from '@/components/searchbox/v2/NodeSearchCategoryTreeNode.vue'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { nodeOrganizationService } from '@/services/nodeOrganizationService'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { NodeSourceType } from '@/types/nodeSource'
import type { TreeNode } from '@/types/treeExplorerTypes'
import { cn } from '@/utils/tailwindUtil'

const selectedCategory = defineModel<string>('selectedCategory', {
  required: true
})

const { t } = useI18n()
const { flags } = useFeatureFlags()
const nodeDefStore = useNodeDefStore()

const topCategories = computed(() => [
  { id: 'most-relevant', label: t('g.mostRelevant') },
  { id: 'favorites', label: t('g.favorites') }
])

const hasEssentialNodes = computed(() =>
  nodeDefStore.visibleNodeDefs.some(
    (n) => n.nodeSource.type === NodeSourceType.Essentials
  )
)

const sourceCategories = computed(() => {
  const categories = []
  if (flags.nodeLibraryEssentialsEnabled && hasEssentialNodes.value) {
    categories.push({ id: 'essentials', label: t('g.essentials') })
  }
  categories.push({ id: 'custom', label: t('g.custom') })
  return categories
})

const categoryTree = computed<CategoryNode[]>(() => {
  const tree = nodeOrganizationService.organizeNodes(
    nodeDefStore.visibleNodeDefs,
    { groupBy: 'category' }
  )

  const stripRootPrefix = (key: string) => key.replace(/^root\//, '')

  function mapNode(node: TreeNode): CategoryNode {
    const children = node.children
      ?.filter((child): child is TreeNode => !child.leaf)
      .map(mapNode)
    return {
      key: stripRootPrefix(node.key as string),
      label: node.label,
      ...(children?.length ? { children } : {})
    }
  }

  return (tree.children ?? [])
    .filter((node): node is TreeNode => !node.leaf)
    .map(mapNode)
})

function categoryBtnClass(id: string) {
  return cn(
    'cursor-pointer border-none bg-transparent rounded px-3 py-2.5 text-left text-sm transition-colors',
    selectedCategory.value === id
      ? CATEGORY_SELECTED_CLASS
      : CATEGORY_UNSELECTED_CLASS
  )
}

const selectedCollapsed = ref(false)

function selectCategory(categoryId: string) {
  if (selectedCategory.value === categoryId) {
    selectedCollapsed.value = !selectedCollapsed.value
  } else {
    selectedCollapsed.value = false
    selectedCategory.value = categoryId
  }
}
</script>
