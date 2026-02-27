<script setup lang="ts">
import type { MaybeRefOrGetter } from 'vue'

import Popover from 'primevue/popover'
import { ref, useTemplateRef } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import type {
  FilterOption,
  OwnershipFilterOption,
  OwnershipOption
} from '@/platform/assets/types/filterTypes'
import { cn } from '@/utils/tailwindUtil'

import FormSearchInput from '../FormSearchInput.vue'
import type { LayoutMode, SortOption } from './types'

const { t } = useI18n()

defineProps<{
  searcher?: (
    query: string,
    onCleanup: (cleanupFn: () => void) => void
  ) => Promise<void>
  sortOptions: SortOption[]
  updateKey?: MaybeRefOrGetter<unknown>
  showOwnershipFilter?: boolean
  ownershipOptions?: OwnershipFilterOption[]
  showBaseModelFilter?: boolean
  baseModelOptions?: FilterOption[]
}>()

const layoutMode = defineModel<LayoutMode>('layoutMode')
const searchQuery = defineModel<string>('searchQuery')
const sortSelected = defineModel<string>('sortSelected')
const ownershipSelected = defineModel<OwnershipOption>('ownershipSelected', {
  default: 'all'
})
const baseModelSelected = defineModel<Set<string>>('baseModelSelected', {
  default: new Set()
})

const actionButtonStyle = cn(
  'h-8 bg-zinc-500/20 rounded-lg outline outline-1 outline-offset-[-1px] outline-node-component-border transition-all duration-150'
)

const layoutSwitchItemStyle =
  'size-6 flex justify-center items-center rounded-sm cursor-pointer transition-all duration-150 hover:scale-108 hover:text-base-foreground active:scale-95'

const sortPopoverRef = useTemplateRef('sortPopoverRef')
const sortTriggerRef = useTemplateRef('sortTriggerRef')
const isSortPopoverOpen = ref(false)

function toggleSortPopover(event: Event) {
  if (!sortPopoverRef.value || !sortTriggerRef.value) return
  isSortPopoverOpen.value = !isSortPopoverOpen.value
  sortPopoverRef.value.toggle(event, sortTriggerRef.value.$el)
}
function closeSortPopover() {
  isSortPopoverOpen.value = false
  sortPopoverRef.value?.hide()
}

function handleSortSelected(item: SortOption) {
  sortSelected.value = item.id
  closeSortPopover()
}

const ownershipPopoverRef = useTemplateRef('ownershipPopoverRef')
const ownershipTriggerRef = useTemplateRef('ownershipTriggerRef')
const isOwnershipPopoverOpen = ref(false)

function toggleOwnershipPopover(event: Event) {
  if (!ownershipPopoverRef.value || !ownershipTriggerRef.value) return
  isOwnershipPopoverOpen.value = !isOwnershipPopoverOpen.value
  ownershipPopoverRef.value.toggle(event, ownershipTriggerRef.value.$el)
}
function closeOwnershipPopover() {
  isOwnershipPopoverOpen.value = false
  ownershipPopoverRef.value?.hide()
}

function handleOwnershipSelected(item: OwnershipFilterOption) {
  ownershipSelected.value = item.value
  closeOwnershipPopover()
}

const baseModelPopoverRef = useTemplateRef('baseModelPopoverRef')
const baseModelTriggerRef = useTemplateRef('baseModelTriggerRef')
const isBaseModelPopoverOpen = ref(false)

function toggleBaseModelPopover(event: Event) {
  if (!baseModelPopoverRef.value || !baseModelTriggerRef.value) return
  isBaseModelPopoverOpen.value = !isBaseModelPopoverOpen.value
  baseModelPopoverRef.value.toggle(event, baseModelTriggerRef.value.$el)
}

function toggleBaseModelSelection(item: FilterOption) {
  const current = new Set(baseModelSelected.value)
  baseModelSelected.value = current.has(item.value)
    ? new Set([...current].filter((v) => v !== item.value))
    : new Set([...current, item.value])
}
</script>

<template>
  <div class="text-secondary flex gap-2 px-4">
    <FormSearchInput
      v-model="searchQuery"
      :searcher
      :update-key
      :class="
        cn(
          actionButtonStyle,
          'hover:outline-component-node-widget-background-highlighted/80',
          'focus-within:outline-component-node-widget-background-highlighted/80 focus-within:ring-0'
        )
      "
    />

    <Button
      ref="sortTriggerRef"
      variant="textonly"
      size="icon"
      :class="
        cn(
          actionButtonStyle,
          'relative w-8 hover:outline-component-node-widget-background-highlighted active:scale-95'
        )
      "
      @click="toggleSortPopover"
    >
      <div
        v-if="sortSelected !== 'default'"
        class="absolute top-[-2px] left-[-2px] size-2 rounded-full bg-component-node-widget-background-highlighted"
      />
      <i class="icon-[lucide--arrow-up-down] size-4" />
    </Button>
    <Popover
      ref="sortPopoverRef"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="{
        root: {
          class: 'absolute z-50'
        },
        content: {
          class: ['bg-transparent border-none p-0 pt-2 rounded-lg shadow-lg']
        }
      }"
      @hide="isSortPopoverOpen = false"
    >
      <div
        :class="
          cn(
            'flex flex-col gap-2 p-2 min-w-32',
            'bg-component-node-background',
            'rounded-lg outline outline-offset-[-1px] outline-component-node-border'
          )
        "
      >
        <Button
          v-for="item of sortOptions"
          :key="item.name"
          variant="textonly"
          size="unset"
          :class="cn('flex justify-between items-center h-6 text-left')"
          @click="handleSortSelected(item)"
        >
          <span>{{ item.name }}</span>
          <i
            v-if="sortSelected === item.id"
            class="icon-[lucide--check] size-4"
          />
        </Button>
      </div>
    </Popover>

    <Button
      v-if="showOwnershipFilter && ownershipOptions?.length"
      ref="ownershipTriggerRef"
      :aria-label="t('assetBrowser.ownership')"
      :title="t('assetBrowser.ownership')"
      variant="textonly"
      size="icon"
      :class="
        cn(
          actionButtonStyle,
          'relative w-8 hover:outline-component-node-widget-background-highlighted active:scale-95'
        )
      "
      @click="toggleOwnershipPopover"
    >
      <div
        v-if="ownershipSelected !== 'all'"
        class="absolute top-[-2px] left-[-2px] size-2 rounded-full bg-component-node-widget-background-highlighted"
      />
      <i class="icon-[lucide--user] size-4" />
    </Button>
    <Popover
      ref="ownershipPopoverRef"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="{
        root: {
          class: 'absolute z-50'
        },
        content: {
          class: ['bg-transparent border-none p-0 pt-2 rounded-lg shadow-lg']
        }
      }"
      @hide="isOwnershipPopoverOpen = false"
    >
      <div
        :class="
          cn(
            'flex flex-col gap-2 p-2 min-w-32',
            'bg-component-node-background',
            'rounded-lg outline outline-offset-[-1px] outline-component-node-border'
          )
        "
      >
        <Button
          v-for="item of ownershipOptions"
          :key="item.value"
          variant="textonly"
          size="unset"
          :class="cn('flex justify-between items-center h-6 text-left')"
          @click="handleOwnershipSelected(item)"
        >
          <span>{{ item.name }}</span>
          <i
            v-if="ownershipSelected === item.value"
            class="icon-[lucide--check] size-4"
          />
        </Button>
      </div>
    </Popover>

    <Button
      v-if="showBaseModelFilter && baseModelOptions?.length"
      ref="baseModelTriggerRef"
      :aria-label="t('assetBrowser.baseModel')"
      :title="t('assetBrowser.baseModel')"
      variant="textonly"
      size="icon"
      :class="
        cn(
          actionButtonStyle,
          'relative w-8 hover:outline-component-node-widget-background-highlighted active:scale-95'
        )
      "
      @click="toggleBaseModelPopover"
    >
      <div
        v-if="baseModelSelected.size > 0"
        class="absolute top-[-2px] left-[-2px] size-2 rounded-full bg-component-node-widget-background-highlighted"
      />
      <i class="icon-[comfy--ai-model] size-4" />
    </Button>
    <Popover
      ref="baseModelPopoverRef"
      :dismissable="true"
      :close-on-escape="true"
      unstyled
      :pt="{
        root: {
          class: 'absolute z-50'
        },
        content: {
          class: ['bg-transparent border-none p-0 pt-2 rounded-lg shadow-lg']
        }
      }"
      @hide="isBaseModelPopoverOpen = false"
    >
      <div
        :class="
          cn(
            'flex flex-col gap-2 p-2 min-w-32',
            'bg-component-node-background',
            'rounded-lg outline outline-offset-[-1px] outline-component-node-border'
          )
        "
      >
        <Button
          v-for="item of baseModelOptions"
          :key="item.value"
          variant="textonly"
          size="unset"
          :class="cn('flex justify-between items-center h-6 text-left')"
          @click="toggleBaseModelSelection(item)"
        >
          <span>{{ item.name }}</span>
          <i
            v-if="baseModelSelected.has(item.value)"
            class="icon-[lucide--check] size-4"
          />
        </Button>
        <span class="h-0 w-full border-b border-border-default" />
        <Button
          variant="textonly"
          size="unset"
          :class="cn('flex justify-between items-center h-6 text-left')"
          @click="baseModelSelected = new Set()"
        >
          {{ t('g.clearFilters') }}
        </Button>
      </div>
    </Popover>

    <div
      :class="
        cn(
          actionButtonStyle,
          'flex justify-center items-center p-1 gap-1 hover:outline-component-node-widget-background-highlighted'
        )
      "
    >
      <Button
        variant="textonly"
        size="unset"
        :class="
          cn(
            layoutSwitchItemStyle,
            layoutMode === 'list' && 'bg-neutral-500/50 text-base-foreground'
          )
        "
        @click="layoutMode = 'list'"
      >
        <i class="icon-[lucide--list] size-4" />
      </Button>
      <Button
        variant="textonly"
        size="unset"
        :class="
          cn(
            layoutSwitchItemStyle,
            layoutMode === 'grid' && 'bg-neutral-500/50 text-base-foreground'
          )
        "
        @click="layoutMode = 'grid'"
      >
        <i class="icon-[lucide--layout-grid] size-4" />
      </Button>
    </div>
  </div>
</template>
