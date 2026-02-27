<template>
  <div>
    <div
      v-tooltip.top="
        isDisabled ? $t('manager.enablePackToChangeVersion') : null
      "
      class="inline-flex items-center gap-1 rounded-2xl py-1 text-xs"
      :class="{
        'bg-dialog-surface px-1.5': fill,
        'cursor-pointer': !isDisabled,
        'cursor-not-allowed opacity-60': isDisabled
      }"
      :aria-haspopup="!isDisabled"
      :role="isDisabled ? 'text' : 'button'"
      :tabindex="isDisabled ? -1 : 0"
      @click="!isDisabled && toggleVersionSelector($event)"
      @keydown.enter="!isDisabled && toggleVersionSelector($event)"
      @keydown.space="!isDisabled && toggleVersionSelector($event)"
    >
      <i
        v-if="isUpdateAvailable"
        class="pi pi-arrow-circle-up text-xs text-blue-600"
      />
      <span>{{ installedVersion }}</span>
      <i v-if="!isDisabled" class="pi pi-chevron-right text-xxs" />
    </div>

    <Popover
      ref="popoverRef"
      append-to="body"
      :pt="{
        root: {
          class: 'before:hidden after:hidden [&_.p-popover-arrow]:hidden'
        },
        content: { class: 'p-0 shadow-lg' }
      }"
      @show="fixPopoverIntoViewport"
    >
      <PackVersionSelectorPopover
        :installed-version="installedVersion"
        :node-pack="nodePack"
        @cancel="closeVersionSelector"
        @submit="closeVersionSelector"
      />
    </Popover>
  </div>
</template>

<script setup lang="ts">
import Popover from 'primevue/popover'
import { valid as validSemver } from 'semver'
import { computed, nextTick, ref, watch } from 'vue'

import type { components } from '@/types/comfyRegistryTypes'
import PackVersionSelectorPopover from '@/workbench/extensions/manager/components/manager/PackVersionSelectorPopover.vue'
import { usePackUpdateStatus } from '@/workbench/extensions/manager/composables/nodePack/usePackUpdateStatus'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

const TRUNCATED_HASH_LENGTH = 7

const {
  nodePack,
  isSelected,
  fill = true
} = defineProps<{
  nodePack: components['schemas']['Node']
  isSelected: boolean
  fill?: boolean
}>()

const { isUpdateAvailable } = usePackUpdateStatus(() => nodePack)
const popoverRef = ref()
const lastTargetEl = ref<HTMLElement | null>(null)

const managerStore = useComfyManagerStore()

const isInstalled = computed(() => managerStore.isPackInstalled(nodePack?.id))
const isDisabled = computed(
  () => isInstalled.value && !managerStore.isPackEnabled(nodePack?.id)
)

const installedVersion = computed(() => {
  if (!nodePack.id) return 'nightly'
  const version =
    managerStore.installedPacks[nodePack.id]?.ver ??
    nodePack.latest_version?.version ??
    'nightly'

  // If Git hash, truncate to 7 characters
  return validSemver(version)
    ? version
    : version.slice(0, TRUNCATED_HASH_LENGTH)
})

const toggleVersionSelector = (event: Event) => {
  lastTargetEl.value = event.currentTarget as HTMLElement
  popoverRef.value.toggle(event)
}

const closeVersionSelector = () => {
  popoverRef.value.hide()
}

const fixPopoverIntoViewport = async () => {
  await nextTick()

  const popoverEl: HTMLElement | undefined =
    popoverRef.value?.container ?? popoverRef.value?.$el
  const targetEl = lastTargetEl.value
  if (!popoverEl || !targetEl) return

  requestAnimationFrame(() => {
    const boundaryEl =
      targetEl.closest('.p-dialog') ??
      targetEl.closest('.manager-dialog') ??
      targetEl.closest('[role="dialog"]') ??
      document.documentElement

    const boundary = boundaryEl.getBoundingClientRect()
    const targetRect = targetEl.getBoundingClientRect()

    popoverEl.style.transform = ''

    const rect = popoverEl.getBoundingClientRect()

    const M = 8 // keep away from dialog edge
    const GAP = 10

    const spaceBelow = boundary.bottom - targetRect.bottom
    const spaceAbove = targetRect.top - boundary.top

    // Decide side (below by default; flip to above if needed)
    const placeAbove = spaceBelow < rect.height + GAP && spaceAbove > spaceBelow

    // 1) Top with a GAP (prevents covering trigger)
    let top = placeAbove
      ? targetRect.top - rect.height - GAP
      : targetRect.bottom + GAP

    // Clamp vertically
    top = Math.min(top, boundary.bottom - rect.height - M)
    top = Math.max(top, boundary.top + M)

    // 2) Left (align to trigger, then clamp)
    let left = targetRect.left
    left = Math.min(left, boundary.right - rect.width - M)
    left = Math.max(left, boundary.left + M)

    // Apply position
    popoverEl.style.top = `${Math.round(top)}px`
    popoverEl.style.left = `${Math.round(left)}px`

    popoverEl.classList.toggle('p-popover-flipped', placeAbove)
  })
}

// If the card is unselected, automatically close the version selector popover
watch(
  () => isSelected,
  (isSelected, wasSelected) => {
    if (wasSelected && !isSelected) {
      closeVersionSelector()
    }
  }
)
</script>
