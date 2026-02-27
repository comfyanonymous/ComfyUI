<script setup lang="ts">
import { useMounted, watchDebounced } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  shallowRef
} from 'vue'
import { useI18n } from 'vue-i18n'

import FormSearchInput from '@/renderer/extensions/vueNodes/widgets/components/form/FormSearchInput.vue'
import { DraggableList } from '@/scripts/ui/draggableList'
import { useFavoritedWidgetsStore } from '@/stores/workspace/favoritedWidgetsStore'
import type { ValidFavoritedWidget } from '@/stores/workspace/favoritedWidgetsStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'

import { searchWidgets } from '../shared'
import SectionWidgets from './SectionWidgets.vue'

const favoritedWidgetsStore = useFavoritedWidgetsStore()
const rightSidePanelStore = useRightSidePanelStore()
const { searchQuery } = storeToRefs(rightSidePanelStore)
const { t } = useI18n()

const draggableList = ref<DraggableList | undefined>(undefined)
const sectionWidgetsRef = ref<{ widgetsContainer: HTMLElement }>()
const isSearching = ref(false)

const favoritedWidgets = computed(
  () => favoritedWidgetsStore.validFavoritedWidgets
)

const label = computed(() =>
  favoritedWidgets.value.length === 0
    ? t('rightSidePanel.favoritesNone')
    : t('rightSidePanel.favorites')
)

const searchedFavoritedWidgets = shallowRef<ValidFavoritedWidget[]>(
  favoritedWidgets.value
)

async function searcher(query: string) {
  isSearching.value = query.trim().length > 0
  searchedFavoritedWidgets.value = searchWidgets(favoritedWidgets.value, query)
}

const isMounted = useMounted()

function setDraggableState() {
  if (!isMounted.value) return
  draggableList.value?.dispose()
  const container = sectionWidgetsRef.value?.widgetsContainer
  if (isSearching.value || !container?.children?.length) return

  draggableList.value = new DraggableList(container, '.draggable-item')

  draggableList.value.applyNewItemsOrder = function () {
    const reorderedItems: HTMLElement[] = []

    let oldPosition = -1
    this.getAllItems().forEach((item, index) => {
      if (item === this.draggableItem) {
        oldPosition = index
        return
      }
      if (!this.isItemToggled(item)) {
        reorderedItems[index] = item
        return
      }
      const newIndex = this.isItemAbove(item) ? index + 1 : index - 1
      reorderedItems[newIndex] = item
    })

    for (let index = 0; index < this.getAllItems().length; index++) {
      const item = reorderedItems[index]
      if (typeof item === 'undefined') {
        reorderedItems[index] = this.draggableItem as HTMLElement
      }
    }

    const newPosition = reorderedItems.indexOf(
      this.draggableItem as HTMLElement
    )
    const widgets = [...searchedFavoritedWidgets.value]
    const [widget] = widgets.splice(oldPosition, 1)
    widgets.splice(newPosition, 0, widget)
    searchedFavoritedWidgets.value = widgets
    favoritedWidgetsStore.reorderFavorites(widgets)
  }
}

watchDebounced(
  searchedFavoritedWidgets,
  () => {
    setDraggableState()
  },
  { debounce: 100 }
)

onMounted(() => {
  setDraggableState()
})

onBeforeUnmount(() => {
  draggableList.value?.dispose()
})
</script>

<template>
  <div class="px-4 pt-1 pb-4 flex gap-2 border-b border-interface-stroke">
    <FormSearchInput
      v-model="searchQuery"
      :searcher
      :update-key="favoritedWidgets"
    />
  </div>
  <SectionWidgets
    ref="sectionWidgetsRef"
    :label
    :widgets="searchedFavoritedWidgets"
    :is-draggable="!isSearching"
    hidden-favorite-indicator
    show-node-name
    enable-empty-state
    class="border-b border-interface-stroke"
    @update:collapse="nextTick(setDraggableState)"
  >
    <template #empty>
      <div class="text-sm text-muted-foreground px-4 text-center py-10">
        <p>
          {{
            isSearching
              ? t('rightSidePanel.noneSearchDesc')
              : t('rightSidePanel.favoritesNoneDesc')
          }}
        </p>
        <i18n-t
          v-if="!isSearching"
          keypath="rightSidePanel.favoritesNoneHint"
          tag="p"
          class="mt-2 text-xs"
        >
          <template #moreIcon>
            <span
              aria-hidden="true"
              class="inline-flex size-5 items-center justify-center rounded-md bg-secondary-background-hover text-secondary-foreground align-middle"
            >
              <i class="icon-[lucide--more-vertical] text-sm" />
            </span>
          </template>
        </i18n-t>
      </div>
    </template>
  </SectionWidgets>
</template>
