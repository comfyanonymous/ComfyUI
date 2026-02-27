<template>
  <SidebarTabTemplate :title="$t('sideToolbar.modelLibrary')">
    <template #tool-buttons>
      <Button
        v-tooltip.bottom="$t('g.refresh')"
        variant="muted-textonly"
        size="icon"
        :aria-label="$t('g.refresh')"
        @click="modelStore.loadModelFolders"
      >
        <i class="icon-[lucide--refresh-cw] size-4" />
      </Button>
      <Button
        v-tooltip.bottom="$t('g.loadAllFolders')"
        variant="muted-textonly"
        size="icon"
        :aria-label="$t('g.loadAllFolders')"
        @click="modelStore.loadModels"
      >
        <i class="icon-[lucide--cloud-download] size-4" />
      </Button>
    </template>
    <template #header>
      <div class="px-2 2xl:px-4">
        <SearchBox
          ref="searchBoxRef"
          v-model:model-value="searchQuery"
          :placeholder="
            $t('g.searchPlaceholder', {
              subject: $t('sideToolbar.labels.models')
            })
          "
          @search="handleSearch"
        />
      </div>
    </template>
    <template #body>
      <ElectronDownloadItems v-if="isDesktop" />

      <Divider type="dashed" class="m-2" />
      <TreeExplorer
        v-model:expanded-keys="expandedKeys"
        class="model-lib-tree-explorer"
        :root="renderedRoot"
      >
        <template #node="{ node }">
          <ModelTreeLeaf :node="node" />
        </template>
      </TreeExplorer>
    </template>
  </SidebarTabTemplate>
  <div id="model-library-model-preview-container" />
</template>

<script setup lang="ts">
import { Divider } from 'primevue'
import { computed, nextTick, onMounted, ref, toRef, watch } from 'vue'

import SearchBox from '@/components/common/SearchBox.vue'
import TreeExplorer from '@/components/common/TreeExplorer.vue'
import SidebarTabTemplate from '@/components/sidebar/tabs/SidebarTabTemplate.vue'
import ElectronDownloadItems from '@/components/sidebar/tabs/modelLibrary/ElectronDownloadItems.vue'
import ModelTreeLeaf from '@/components/sidebar/tabs/modelLibrary/ModelTreeLeaf.vue'
import Button from '@/components/ui/button/Button.vue'
import { useTreeExpansion } from '@/composables/useTreeExpansion'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useLitegraphService } from '@/services/litegraphService'
import type { ComfyModelDef, ModelFolder } from '@/stores/modelStore'
import { ResourceState, useModelStore } from '@/stores/modelStore'
import { useModelToNodeStore } from '@/stores/modelToNodeStore'
import type { TreeExplorerNode, TreeNode } from '@/types/treeExplorerTypes'
import { isDesktop } from '@/platform/distribution/types'
import { buildTree } from '@/utils/treeUtil'

const modelStore = useModelStore()
const modelToNodeStore = useModelToNodeStore()
const settingStore = useSettingStore()
const searchBoxRef = ref()
const searchQuery = ref<string>('')
const expandedKeys = ref<Record<string, boolean>>({})
const { expandNode, toggleNodeOnEvent } = useTreeExpansion(expandedKeys)

const filteredModels = ref<ComfyModelDef[]>([])
const handleSearch = async (query: string) => {
  if (!query) {
    filteredModels.value = []
    expandedKeys.value = {}
    return
  }
  // Load all models to ensure we have the latest data
  await modelStore.loadModels()
  const search = query.toLocaleLowerCase()
  filteredModels.value = modelStore.models.filter((model: ComfyModelDef) => {
    return model.searchable.includes(search)
  })

  await nextTick()
  expandNode(root.value)
}

type ModelOrFolder = ComfyModelDef | ModelFolder

const root = computed<TreeNode>(() => {
  const allNodes: ModelOrFolder[] = searchQuery.value
    ? filteredModels.value
    : [...modelStore.modelFolders, ...modelStore.models]
  return buildTree(allNodes, (modelOrFolder: ModelOrFolder) =>
    modelOrFolder.key.split('/')
  )
})

const renderedRoot = computed<TreeExplorerNode<ModelOrFolder>>(() => {
  const nameFormat = settingStore.get('Comfy.ModelLibrary.NameFormat')
  const fillNodeInfo = (node: TreeNode): TreeExplorerNode<ModelOrFolder> => {
    const children = node.children?.map(fillNodeInfo)
    const model: ComfyModelDef | null =
      node.leaf && node.data ? node.data : null
    const folder: ModelFolder | null =
      !node.leaf && node.data ? node.data : null

    return {
      key: node.key,
      label: model
        ? nameFormat === 'title'
          ? model.title
          : model.simplified_file_name
        : node.label,
      leaf: node.leaf,
      data: node.data,
      getIcon() {
        if (model) {
          return model.image ? 'pi pi-image' : 'pi pi-file'
        }
        if (folder) {
          return folder.state === ResourceState.Loading
            ? 'pi pi-spin pi-spinner'
            : 'pi pi-folder'
        }
        return 'pi pi-folder'
      },
      getBadgeText() {
        // Return undefined to apply default badge text
        // Return empty string to hide badge
        if (!folder) {
          return
        }
        return folder.state === ResourceState.Loaded ? undefined : ''
      },
      children,
      draggable: node.leaf,
      handleClick(e: MouseEvent) {
        if (this.leaf) {
          // @ts-expect-error fixme ts strict error
          const provider = modelToNodeStore.getNodeProvider(model.directory)
          if (provider) {
            const node = useLitegraphService().addNodeOnGraph(provider.nodeDef)
            // @ts-expect-error fixme ts strict error
            const widget = node.widgets.find(
              (widget) => widget.name === provider.key
            )
            if (widget) {
              // @ts-expect-error fixme ts strict error
              widget.value = model.file_name
            }
          }
        } else {
          toggleNodeOnEvent(e, node)
        }
      }
    }
  }

  return fillNodeInfo(root.value)
})

watch(
  toRef(expandedKeys, 'value'),
  (newExpandedKeys) => {
    Object.entries(newExpandedKeys).forEach(([key, isExpanded]) => {
      if (isExpanded) {
        const folderPath = key.split('/').slice(1).join('/')
        if (folderPath && !folderPath.includes('/')) {
          // Trigger (async) load of model data for this folder
          void modelStore.getLoadedModelFolder(folderPath)
        }
      }
    })
  },
  { deep: true }
)

onMounted(async () => {
  searchBoxRef.value?.focus()
  if (settingStore.get('Comfy.ModelLibrary.AutoLoadAll')) {
    await modelStore.loadModels()
  }
})
</script>

<style scoped>
:deep(.pi-fake-spacer) {
  height: 1px;
  width: 16px;
}
</style>
