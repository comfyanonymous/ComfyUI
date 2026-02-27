<template>
  <BaseModalLayout
    v-model:right-panel-open="isRightPanelOpen"
    :content-title="$t('manager.discoverCommunityContent')"
    :right-panel-title="$t('manager.nodePackInfo')"
    class="manager-dialog"
  >
    <template #leftPanelHeaderTitle>
      <i class="icon-[comfy--extensions-blocks]" />
      <h2 class="text-neutral text-base">{{ $t('manager.title') }}</h2>
    </template>
    <template #leftPanel>
      <LeftSidePanel v-model="selectedNavId" :nav-items="navItems" />
    </template>

    <template #header>
      <div class="flex w-full items-center justify-between gap-2">
        <div class="flex items-center gap-2">
          <SingleSelect
            v-model="searchMode"
            class="min-w-34"
            :options="filterOptions"
          />
          <AutoCompletePlus
            v-model.lazy="searchQuery"
            :suggestions="suggestions"
            :placeholder="$t('manager.searchPlaceholder')"
            :complete-on-focus="false"
            :delay="8"
            option-label="query"
            class="w-full min-w-md max-w-lg"
            :pt="{
              root: { class: 'relative' },
              pcInputText: {
                root: {
                  autofocus: true,
                  class:
                    'w-full h-10 rounded-lg bg-comfy-input text-comfy-input-foreground border-none outline-none text-sm'
                }
              },
              overlay: {
                class:
                  'bg-comfy-input rounded-lg mt-1 shadow-lg border border-border-default'
              },
              list: { class: 'p-1' },
              option: {
                class:
                  'px-3 py-2 rounded hover:bg-button-hover-surface cursor-pointer text-sm'
              },
              loader: { style: 'display: none' }
            }"
            :show-empty-message="false"
            @complete="stubTrue"
            @option-select="onOptionSelect"
          >
            <template #dropdownicon>
              <i
                class="pi pi-search absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
              />
            </template>
          </AutoCompletePlus>
        </div>
        <PackInstallButton
          v-if="isMissingTab && missingNodePacks.length > 0"
          :disabled="isMissingLoading || !!missingError"
          :node-packs="missingNodePacks"
          size="lg"
          :label="$t('manager.installAllMissingNodes')"
        />
        <PackUpdateButton
          v-if="isUpdateAvailableTab && hasUpdateAvailable"
          :node-packs="enabledUpdateAvailableNodePacks"
          :has-disabled-update-packs="hasDisabledUpdatePacks"
          size="lg"
        />
      </div>
    </template>

    <template #contentFilter>
      <!-- Conflict Warning Banner -->
      <div
        v-if="shouldShowManagerBanner"
        class="relative mx-6 mt-3 mb-4 flex items-center gap-6 rounded-lg bg-yellow-500/20 p-4"
      >
        <i
          class="icon-[lucide--triangle-alert] text-lg text-warning-background"
        />
        <div class="flex flex-1 flex-col gap-2">
          <p class="m-0 text-sm font-bold">
            {{ $t('manager.conflicts.warningBanner.title') }}
          </p>
          <p class="m-0 text-xs">
            {{ $t('manager.conflicts.warningBanner.message') }}
          </p>
          <p
            class="m-0 cursor-pointer text-sm font-bold"
            @click="onClickWarningLink"
          >
            {{ $t('manager.conflicts.warningBanner.button') }}
          </p>
        </div>
        <Button
          class="absolute top-0 right-0"
          variant="textonly"
          size="icon"
          @click="dismissWarningBanner"
        >
          <i class="pi pi-times text-xs text-base-foreground"></i>
        </Button>
      </div>

      <!-- Sort Options -->
      <div class="flex justify-end px-6 pb-4">
        <SingleSelect
          v-model="sortField"
          :label="$t('g.sort')"
          :options="availableSortOptions"
          class="w-48"
        >
          <template #icon>
            <i class="icon-[lucide--arrow-up-down] text-muted-foreground" />
          </template>
        </SingleSelect>
      </div>
    </template>

    <template #content>
      <div v-if="isLoading" class="scrollbar-hide h-full w-full overflow-auto">
        <GridSkeleton :grid-style="GRID_STYLE" :skeleton-card-count />
      </div>
      <NoResultsPlaceholder
        v-else-if="displayPacks.length === 0"
        :title="emptyStateTitle"
        :message="emptyStateMessage"
      />
      <div v-else class="h-full w-full" @click="handleGridContainerClick">
        <VirtualGrid
          id="results-grid"
          :items="resultsWithKeys"
          :buffer-rows="4"
          :grid-style="GRID_STYLE"
          @approach-end="onApproachEnd"
        >
          <template #item="{ item }">
            <PackCard
              :node-pack="item"
              :is-selected="
                selectedNodePacks.some((pack) => pack.id === item.id)
              "
              @click.stop="(event: MouseEvent) => selectNodePack(item, event)"
            />
          </template>
        </VirtualGrid>
      </div>
    </template>

    <template #rightPanel>
      <InfoPanel
        v-if="!hasMultipleSelections && selectedNodePack"
        :node-pack="selectedNodePack"
      />
      <InfoPanelMultiItem v-else :node-packs="selectedNodePacks" />
    </template>
  </BaseModalLayout>
</template>

<script setup lang="ts">
import { until, whenever } from '@vueuse/core'
import { merge, stubTrue } from 'es-toolkit/compat'
import type { AutoCompleteOptionSelectEvent } from 'primevue/autocomplete'
import {
  computed,
  onBeforeUnmount,
  onMounted,
  onUnmounted,
  provide,
  ref,
  watch,
  watchEffect
} from 'vue'
import { useI18n } from 'vue-i18n'

import NoResultsPlaceholder from '@/components/common/NoResultsPlaceholder.vue'
import VirtualGrid from '@/components/common/VirtualGrid.vue'
import SingleSelect from '@/components/input/SingleSelect.vue'
import AutoCompletePlus from '@/components/primevueOverride/AutoCompletePlus.vue'
import Button from '@/components/ui/button/Button.vue'
import BaseModalLayout from '@/components/widget/layout/BaseModalLayout.vue'
import LeftSidePanel from '@/components/widget/panel/LeftSidePanel.vue'
import { useExternalLink } from '@/composables/useExternalLink'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useComfyRegistryStore } from '@/stores/comfyRegistryStore'
import type { components } from '@/types/comfyRegistryTypes'
import type { NavGroupData, NavItemData } from '@/types/navTypes'
import { OnCloseKey } from '@/types/widgetTypes'
import PackInstallButton from '@/workbench/extensions/manager/components/manager/button/PackInstallButton.vue'
import PackUpdateButton from '@/workbench/extensions/manager/components/manager/button/PackUpdateButton.vue'
import InfoPanel from '@/workbench/extensions/manager/components/manager/infoPanel/InfoPanel.vue'
import InfoPanelMultiItem from '@/workbench/extensions/manager/components/manager/infoPanel/InfoPanelMultiItem.vue'
import PackCard from '@/workbench/extensions/manager/components/manager/packCard/PackCard.vue'
import GridSkeleton from '@/workbench/extensions/manager/components/manager/skeleton/GridSkeleton.vue'
import { useMissingNodes } from '@/workbench/extensions/manager/composables/nodePack/useMissingNodes'
import { useUpdateAvailableNodes } from '@/workbench/extensions/manager/composables/nodePack/useUpdateAvailableNodes'
import { useConflictAcknowledgment } from '@/workbench/extensions/manager/composables/useConflictAcknowledgment'
import { useManagerDisplayPacks } from '@/workbench/extensions/manager/composables/useManagerDisplayPacks'
import { useManagerStatePersistence } from '@/workbench/extensions/manager/composables/useManagerStatePersistence'
import { useRegistrySearch } from '@/workbench/extensions/manager/composables/useRegistrySearch'
import { useConflictDetectionStore } from '@/workbench/extensions/manager/stores/conflictDetectionStore'
import { useLegacySearchTip } from '@/workbench/extensions/manager/composables/useLegacySearchTip'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'

const { initialTab, initialPackId, onClose } = defineProps<{
  initialTab?: ManagerTab
  initialPackId?: string
  onClose: () => void
}>()

provide(OnCloseKey, onClose)

const { t } = useI18n()
const { buildDocsUrl } = useExternalLink()
const comfyManagerStore = useComfyManagerStore()
const { getPackById } = useComfyRegistryStore()
const conflictAcknowledgment = useConflictAcknowledgment()
const conflictDetectionStore = useConflictDetectionStore()
const { isNewManagerUI } = useManagerState()
const workflowStore = useWorkflowStore()
const persistedState = useManagerStatePersistence()
const initialState = persistedState.loadStoredState()

const GRID_STYLE = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(14rem, 1fr))',
  gap: '1rem',
  padding: '0.5rem'
} as const

const {
  shouldShowManagerBanner,
  dismissWarningBanner,
  dismissRedDotNotification
} = conflictAcknowledgment

// Missing nodes composable
const {
  missingNodePacks,
  isLoading: isMissingLoading,
  error: missingError
} = useMissingNodes()

// Update available nodes composable
const {
  hasUpdateAvailable,
  enabledUpdateAvailableNodePacks,
  hasDisabledUpdatePacks
} = useUpdateAvailableNodes()

// Get the current workflow name for the nav item
const workflowName = computed(
  () => workflowStore.activeWorkflow?.filename ?? t('manager.inWorkflow')
)

// Navigation items for LeftSidePanel
const navItems = computed<(NavItemData | NavGroupData)[]>(() => [
  {
    id: ManagerTab.All,
    label: t('manager.nav.allExtensions'),
    icon: 'icon-[lucide--list]'
  },
  {
    id: ManagerTab.NotInstalled,
    label: t('manager.nav.notInstalled'),
    icon: 'icon-[lucide--globe]'
  },
  {
    title: t('manager.nav.installedSection'),
    items: [
      {
        id: ManagerTab.AllInstalled,
        label: t('manager.nav.allInstalled'),
        icon: 'icon-[lucide--download]'
      },
      {
        id: ManagerTab.UpdateAvailable,
        label: t('manager.nav.updatesAvailable'),
        icon: 'icon-[lucide--refresh-cw]'
      },
      {
        id: ManagerTab.Conflicting,
        label: t('manager.nav.conflicting'),
        icon: 'icon-[lucide--triangle-alert]',
        badge: conflictDetectionStore.conflictedPackages.length || undefined
      }
    ]
  },
  {
    title: t('manager.nav.inWorkflowSection'),
    items: [
      {
        id: ManagerTab.Workflow,
        label: t('manager.nav.allInWorkflow', {
          workflowName: workflowName.value
        }),
        icon: 'icon-[lucide--share-2]'
      },
      {
        id: ManagerTab.Missing,
        label: t('manager.nav.missingNodes'),
        icon: 'icon-[lucide--triangle-alert]'
      }
    ]
  }
])

const initialTabId = initialTab ?? initialState.selectedTabId ?? ManagerTab.All
const selectedNavId = ref<string | null>(initialTabId)

// Helper to find a nav item by id in the nested structure
const findNavItemById = (
  items: (NavItemData | NavGroupData)[],
  id: string | null
): NavItemData | undefined => {
  for (const item of items) {
    if ('items' in item) {
      const found = item.items.find((subItem) => subItem.id === id)
      if (found) return found
    } else if (item.id === id) {
      return item
    }
  }
  return undefined
}

const selectedTab = computed(() =>
  findNavItemById(navItems.value, selectedNavId.value)
)

const {
  searchQuery,
  pageNumber,
  isLoading: isSearchLoading,
  searchResults,
  searchMode,
  sortField,
  suggestions,
  sortOptions
} = useRegistrySearch({
  initialSortField: initialState.sortField,
  initialSearchMode:
    initialPackId && initialTabId !== ManagerTab.Missing
      ? 'packs'
      : initialState.searchMode,
  initialSearchQuery:
    initialTabId === ManagerTab.Missing
      ? ''
      : (initialPackId ?? initialState.searchQuery)
})
pageNumber.value = 0

const { isLegacyManagerSearch } = useLegacySearchTip(
  searchQuery,
  isNewManagerUI
)

const filterOptions = computed(() => [
  { name: t('manager.filter.nodePack'), value: 'packs' },
  { name: t('g.nodes'), value: 'nodes' }
])

const availableSortOptions = computed(() => {
  if (!sortOptions.value) return []
  return sortOptions.value.map((field) => ({
    name: field.label,
    value: field.id
  }))
})

const onOptionSelect = (event: AutoCompleteOptionSelectEvent) => {
  searchQuery.value = event.value.query
}

const onApproachEnd = () => {
  pageNumber.value++
}

const isInitialLoad = computed(
  () => searchResults.value.length === 0 && searchQuery.value === ''
)

// Use the new composable for tab-based display packs
const {
  displayPacks,
  isLoading: isTabLoading,
  workflowPacks
} = useManagerDisplayPacks(selectedNavId, searchResults, searchQuery, sortField)

// Tab helpers for template
const isUpdateAvailableTab = computed(
  () => selectedTab.value?.id === ManagerTab.UpdateAvailable
)
const isMissingTab = computed(
  () => selectedTab.value?.id === ManagerTab.Missing
)

// Map of tab IDs to their empty state i18n key suffixes
const tabEmptyStateKeys: Partial<Record<ManagerTab, string>> = {
  [ManagerTab.AllInstalled]: 'allInstalled',
  [ManagerTab.UpdateAvailable]: 'updateAvailable',
  [ManagerTab.Conflicting]: 'conflicting',
  [ManagerTab.Workflow]: 'workflow',
  [ManagerTab.Missing]: 'missing'
}

// Empty state messages based on current tab and search state
const emptyStateTitle = computed(() => {
  if (comfyManagerStore.error) return t('manager.errorConnecting')
  if (searchQuery.value) return t('manager.noResultsFound')

  const tabId = selectedTab.value?.id as ManagerTab | undefined
  const emptyStateKey = tabId ? tabEmptyStateKeys[tabId] : undefined

  return emptyStateKey
    ? t(`manager.emptyState.${emptyStateKey}.title`)
    : t('manager.noResultsFound')
})

const emptyStateMessage = computed(() => {
  if (comfyManagerStore.error) return t('manager.tryAgainLater')
  if (searchQuery.value) {
    const baseMessage = t('manager.tryDifferentSearch')
    if (isLegacyManagerSearch.value) {
      return `${baseMessage}\n\n${t('manager.legacyManagerSearchTip')}`
    }
    return baseMessage
  }

  const tabId = selectedTab.value?.id as ManagerTab | undefined
  const emptyStateKey = tabId ? tabEmptyStateKeys[tabId] : undefined

  return emptyStateKey
    ? t(`manager.emptyState.${emptyStateKey}.message`)
    : t('manager.tryDifferentSearch')
})

const onClickWarningLink = () => {
  window.open(
    buildDocsUrl('/troubleshooting/custom-node-issues', {
      includeLocale: true
    }),
    '_blank'
  )
}

const isLoading = computed(() => {
  if (isSearchLoading.value) return searchResults.value.length === 0
  if (isTabLoading.value) return true
  return isInitialLoad.value
})

const resultsWithKeys = computed(
  () =>
    displayPacks.value.map((item) => ({
      ...item,
      key: item.id || item.name
    })) as (components['schemas']['Node'] & { key: string })[]
)

const selectedNodePacks = ref<components['schemas']['Node'][]>([])
const selectedNodePack = computed<components['schemas']['Node'] | null>(() =>
  selectedNodePacks.value.length === 1 ? selectedNodePacks.value[0] : null
)
const isRightPanelOpen = ref(false)

watch(
  () => selectedNodePacks.value.length,
  (length, oldLength) => {
    if (length > 0 && oldLength === 0) {
      isRightPanelOpen.value = true
    }
  }
)

// Auto-select the pack matching initialPackId once
if (initialPackId) {
  until(resultsWithKeys)
    .toMatch((packs) => packs.some((p) => p.id === initialPackId))
    .then((packs) => {
      const target = packs.find((p) => p.id === initialPackId)
      if (target && selectedNodePacks.value.length === 0) {
        selectedNodePacks.value = [target]
        isRightPanelOpen.value = true
      }
    })
}

const getLoadingCount = () => {
  switch (selectedTab.value?.id) {
    case ManagerTab.AllInstalled:
      return comfyManagerStore.installedPacksIds?.size
    case ManagerTab.Workflow:
      return workflowPacks.value?.length
    case ManagerTab.Missing:
      return workflowPacks.value?.filter?.(
        (pack) => !comfyManagerStore.isPackInstalled(pack.id)
      )?.length
    default:
      return searchResults.value.length
  }
}

const skeletonCardCount = computed(() => {
  const loadingCount = getLoadingCount()
  if (loadingCount) return loadingCount
  return 16
})

const selectNodePack = (
  nodePack: components['schemas']['Node'],
  event: MouseEvent
) => {
  if (event.shiftKey || event.ctrlKey || event.metaKey) {
    const index = selectedNodePacks.value.findIndex(
      (pack) => pack.id === nodePack.id
    )

    if (index === -1) {
      selectedNodePacks.value = [...selectedNodePacks.value, nodePack]
    } else {
      selectedNodePacks.value = selectedNodePacks.value.filter(
        (pack) => pack.id !== nodePack.id
      )
    }
  } else {
    selectedNodePacks.value = [nodePack]
  }
}

const unSelectItems = () => {
  selectedNodePacks.value = []
}
const handleGridContainerClick = (event: MouseEvent) => {
  const targetElement = event.target as HTMLElement
  if (targetElement && !targetElement.closest('[data-virtual-grid-item]')) {
    unSelectItems()
  }
}

const hasMultipleSelections = computed(() => selectedNodePacks.value.length > 1)

const lastFetchedPackId = ref<string | null>(null)

whenever(selectedNodePack, async () => {
  getPackById.cancel()
  const pack = selectedNodePack.value
  if (!pack?.id) return
  if (hasMultipleSelections.value) return
  if (lastFetchedPackId.value === pack.id) return
  const data = await getPackById.call(pack.id)
  if (data?.id === pack.id) {
    lastFetchedPackId.value = pack.id
    const mergedPack = merge({}, pack, data)
    const packIndex = selectedNodePacks.value.findIndex(
      (p) => p.id === mergedPack.id
    )
    if (packIndex !== -1) {
      selectedNodePacks.value.splice(packIndex, 1, mergedPack)
    }
  }
})

let gridContainer: HTMLElement | null = null
onMounted(() => {
  gridContainer = document.getElementById('results-grid')
})
watch([searchQuery, selectedNavId], () => {
  gridContainer ??= document.getElementById('results-grid')
  if (gridContainer) {
    pageNumber.value = 0
    gridContainer.scrollTop = 0
  }
  unSelectItems()
})

watchEffect(() => {
  dismissRedDotNotification()
})

onBeforeUnmount(() => {
  persistedState.persistState({
    selectedTabId: (selectedTab.value?.id as ManagerTab) ?? ManagerTab.All,
    searchQuery: searchQuery.value,
    searchMode: searchMode.value,
    sortField: sortField.value
  })
})

onUnmounted(() => {
  getPackById.cancel()
})
</script>
