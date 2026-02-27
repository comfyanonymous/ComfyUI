<template>
  <BaseModalLayout content-title="" data-testid="settings-dialog" size="md">
    <template #leftPanelHeaderTitle>
      <i class="icon-[lucide--settings]" />
      <h2 class="text-neutral text-base">{{ $t('g.settings') }}</h2>
    </template>

    <template #leftPanel>
      <div class="px-3">
        <SearchBox
          v-model:model-value="searchQuery"
          size="md"
          :placeholder="$t('g.searchSettings') + '...'"
          :debounce-time="128"
          @search="handleSearch"
        />
      </div>

      <nav
        ref="navRef"
        class="scrollbar-hide flex flex-1 flex-col gap-1 overflow-y-auto px-3 py-4"
      >
        <div
          v-for="(group, index) in navGroups"
          :key="index"
          class="flex flex-col gap-2"
        >
          <NavTitle :title="group.title" />
          <NavItem
            v-for="item in group.items"
            :key="item.id"
            :data-nav-id="item.id"
            :icon="item.icon"
            :badge="item.badge"
            :active="activeCategoryKey === item.id"
            @click="onNavItemClick(item.id)"
          >
            {{ item.label }}
          </NavItem>
        </div>
      </nav>
    </template>

    <template #header />

    <template #content>
      <template v-if="activePanel">
        <Suspense>
          <component :is="activePanel.component" v-bind="activePanel.props" />
          <template #fallback>
            <div>
              {{ $t('g.loadingPanel', { panel: activePanel.node.label }) }}
            </div>
          </template>
        </Suspense>
      </template>
      <template v-else-if="inSearch">
        <SettingsPanel :setting-groups="searchResults" />
      </template>
      <template v-else-if="activeSettingCategory">
        <CurrentUserMessage v-if="activeSettingCategory.label === 'Comfy'" />
        <ColorPaletteMessage
          v-if="activeSettingCategory.label === 'Appearance'"
        />
        <SettingsPanel :setting-groups="sortedGroups(activeSettingCategory)" />
      </template>
    </template>
  </BaseModalLayout>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, provide, ref, watch } from 'vue'

import SearchBox from '@/components/common/SearchBox.vue'
import CurrentUserMessage from '@/components/dialog/content/setting/CurrentUserMessage.vue'
import BaseModalLayout from '@/components/widget/layout/BaseModalLayout.vue'
import NavItem from '@/components/widget/nav/NavItem.vue'
import NavTitle from '@/components/widget/nav/NavTitle.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import ColorPaletteMessage from '@/platform/settings/components/ColorPaletteMessage.vue'
import SettingsPanel from '@/platform/settings/components/SettingsPanel.vue'
import { useSettingSearch } from '@/platform/settings/composables/useSettingSearch'
import { useSettingUI } from '@/platform/settings/composables/useSettingUI'
import type { SettingTreeNode } from '@/platform/settings/settingStore'
import type {
  ISettingGroup,
  SettingPanelType,
  SettingParams
} from '@/platform/settings/types'
import { OnCloseKey } from '@/types/widgetTypes'
import { flattenTree } from '@/utils/treeUtil'

const { onClose, defaultPanel, scrollToSettingId } = defineProps<{
  onClose: () => void
  defaultPanel?: SettingPanelType
  scrollToSettingId?: string
}>()

provide(OnCloseKey, onClose)

const {
  defaultCategory,
  settingCategories,
  navGroups,
  findCategoryByKey,
  findPanelByKey
} = useSettingUI(defaultPanel, scrollToSettingId)

const {
  searchQuery,
  inSearch,
  searchResultsCategories,
  matchedNavItemKeys,
  handleSearch: handleSearchBase,
  getSearchResults
} = useSettingSearch()

const authActions = useFirebaseAuthActions()

const navRef = ref<HTMLElement | null>(null)
const activeCategoryKey = ref<string | null>(defaultCategory.value?.key ?? null)

const searchableNavItems = computed(() =>
  navGroups.value.flatMap((g) =>
    g.items.map((item) => ({
      key: item.id,
      label: item.label
    }))
  )
)

watch(
  [searchResultsCategories, matchedNavItemKeys],
  ([categories, navKeys]) => {
    if (!inSearch.value || (categories.size === 0 && navKeys.size === 0)) return
    const firstMatch = navGroups.value
      .flatMap((g) => g.items)
      .find((item) => {
        if (navKeys.has(item.id)) return true
        const node = findCategoryByKey(item.id)
        return node && categories.has(node.label)
      })
    activeCategoryKey.value = firstMatch?.id ?? null
  }
)

const activeSettingCategory = computed<SettingTreeNode | null>(() => {
  if (!activeCategoryKey.value) return null
  return (
    settingCategories.value.find((c) => c.key === activeCategoryKey.value) ??
    null
  )
})

const activePanel = computed(() => {
  if (!activeCategoryKey.value) return null
  return findPanelByKey(activeCategoryKey.value)
})

const getGroupSortOrder = (group: SettingTreeNode): number =>
  Math.max(0, ...flattenTree<SettingParams>(group).map((s) => s.sortOrder ?? 0))

function sortedGroups(category: SettingTreeNode): ISettingGroup[] {
  return [...(category.children ?? [])]
    .sort((a, b) => {
      const orderDiff = getGroupSortOrder(b) - getGroupSortOrder(a)
      return orderDiff !== 0 ? orderDiff : a.label.localeCompare(b.label)
    })
    .map((group) => ({
      label: group.label,
      settings: flattenTree<SettingParams>(group).sort((a, b) => {
        const sortOrderA = a.sortOrder ?? 0
        const sortOrderB = b.sortOrder ?? 0
        return sortOrderB - sortOrderA
      })
    }))
}

function handleSearch(query: string) {
  handleSearchBase(query.trim(), searchableNavItems.value)
  if (query) {
    activeCategoryKey.value = null
  } else if (!activeCategoryKey.value) {
    activeCategoryKey.value = defaultCategory.value?.key ?? null
  }
}

function onNavItemClick(id: string) {
  activeCategoryKey.value = id
}

const searchResults = computed<ISettingGroup[]>(() => getSearchResults(null))

// Scroll to and highlight the target setting once the correct category renders.
if (scrollToSettingId) {
  const stopScrollWatch = watch(
    activeCategoryKey,
    () => {
      void nextTick(() => {
        const el = document.querySelector(
          `[data-setting-id="${CSS.escape(scrollToSettingId)}"]`
        )
        if (!el) return
        stopScrollWatch()
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('setting-highlight')
        el.addEventListener(
          'animationend',
          () => el.classList.remove('setting-highlight'),
          { once: true }
        )
      })
    },
    { immediate: true }
  )
  onBeforeUnmount(stopScrollWatch)
}

watch(activeCategoryKey, (newKey, oldKey) => {
  if (!newKey && !inSearch.value) {
    activeCategoryKey.value = oldKey
  }
  if (newKey === 'credits') {
    void authActions.fetchBalance()
  }
  if (newKey) {
    void nextTick(() => {
      navRef.value
        ?.querySelector(`[data-nav-id="${newKey}"]`)
        ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    })
  }
})
</script>

<style>
@media (prefers-reduced-motion: no-preference) {
  .setting-highlight {
    animation: setting-highlight-pulse 1.5s ease-in-out;
  }
}

@keyframes setting-highlight-pulse {
  0%,
  100% {
    background-color: transparent;
  }
  30% {
    background-color: color-mix(
      in srgb,
      var(--p-primary-color) 15%,
      transparent
    );
  }
}
</style>
