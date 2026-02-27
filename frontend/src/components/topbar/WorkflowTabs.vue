<template>
  <div
    ref="containerRef"
    class="workflow-tabs-container flex h-full max-w-full flex-auto flex-row overflow-hidden"
    :class="{ 'workflow-tabs-container-desktop': isDesktop }"
  >
    <Button
      v-if="showOverflowArrows"
      variant="muted-textonly"
      size="icon"
      class="overflow-arrow overflow-arrow-left h-full w-auto aspect-square"
      :aria-label="$t('g.scrollLeft')"
      :disabled="!leftArrowEnabled"
      @mousedown="whileMouseDown($event, () => scroll(-1))"
    >
      <i class="icon-[lucide--chevron-left] size-full" />
    </Button>
    <ScrollPanel
      class="no-drag overflow-hidden"
      :pt:content="{
        class: 'p-0 w-full flex',
        onwheel: handleWheel
      }"
      pt:bar-x="h-1"
    >
      <SelectButton
        class="workflow-tabs bg-transparent"
        :class="props.class"
        :model-value="selectedWorkflow"
        :options="options"
        option-label="label"
        data-key="value"
        @update:model-value="onWorkflowChange"
      >
        <template #option="{ option, index }">
          <WorkflowTab
            :workflow-option="option"
            :is-first="index === 0"
            :is-last="index === options.length - 1"
            @click.middle="onCloseWorkflow(option)"
            @close-to-left="closeWorkflows(options.slice(0, index))"
            @close-to-right="closeWorkflows(options.slice(index + 1))"
            @close-others="
              closeWorkflows([
                ...options.slice(index + 1),
                ...options.slice(0, index)
              ])
            "
          />
        </template>
      </SelectButton>
    </ScrollPanel>
    <Button
      v-if="showOverflowArrows"
      variant="muted-textonly"
      size="icon"
      class="overflow-arrow overflow-arrow-right h-full w-auto aspect-square"
      :aria-label="$t('g.scrollRight')"
      :disabled="!rightArrowEnabled"
      @mousedown="whileMouseDown($event, () => scroll(1))"
    >
      <i class="icon-[lucide--chevron-right] size-full" />
    </Button>
    <WorkflowOverflowMenu
      v-if="showOverflowArrows"
      :workflows="workflowStore.openWorkflows"
      :active-workflow="workflowStore.activeWorkflow"
    />
    <Button
      v-tooltip="{
        value: $t('sideToolbar.newBlankWorkflow'),
        showDelay: 300
      }"
      class="new-blank-workflow-button no-drag shrink-0 rounded-none h-full w-auto aspect-square"
      variant="muted-textonly"
      size="icon"
      :aria-label="$t('sideToolbar.newBlankWorkflow')"
      @click="() => commandStore.execute('Comfy.NewBlankWorkflow')"
    >
      <i class="pi pi-plus" />
    </Button>
    <div
      v-if="isIntegratedTabBar"
      class="ml-auto flex shrink-0 items-center gap-2 px-2"
    >
      <TopMenuHelpButton />
      <CurrentUserButton
        v-if="isLoggedIn"
        :show-arrow="false"
        compact
        class="shrink-0 p-1"
      />
      <LoginButton v-else-if="isDesktop" class="p-1" />
    </div>
    <div v-if="isDesktop" class="window-actions-spacer app-drag shrink-0" />
  </div>
</template>

<script setup lang="ts">
import { useScroll } from '@vueuse/core'
import ScrollPanel from 'primevue/scrollpanel'
import SelectButton from 'primevue/selectbutton'
import { computed, nextTick, onUpdated, ref, watch } from 'vue'
import type { WatchStopHandle } from 'vue'

import CurrentUserButton from '@/components/topbar/CurrentUserButton.vue'
import LoginButton from '@/components/topbar/LoginButton.vue'
import TopMenuHelpButton from '@/components/topbar/TopMenuHelpButton.vue'
import WorkflowTab from '@/components/topbar/WorkflowTab.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useOverflowObserver } from '@/composables/element/useOverflowObserver'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCommandStore } from '@/stores/commandStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import { isDesktop } from '@/platform/distribution/types'
import { whileMouseDown } from '@/utils/mouseDownUtil'

import WorkflowOverflowMenu from './WorkflowOverflowMenu.vue'

interface WorkflowOption {
  value: string
  workflow: ComfyWorkflow
}

const props = defineProps<{
  class?: string
}>()

const settingStore = useSettingStore()
const workspaceStore = useWorkspaceStore()
const workflowStore = useWorkflowStore()
const workflowService = useWorkflowService()
const commandStore = useCommandStore()
const { isLoggedIn } = useCurrentUser()

const isIntegratedTabBar = computed(
  () => settingStore.get('Comfy.UI.TabBarLayout') === 'Integrated'
)

const containerRef = ref<HTMLElement | null>(null)
const showOverflowArrows = ref(false)
const leftArrowEnabled = ref(false)
const rightArrowEnabled = ref(false)

const workflowToOption = (workflow: ComfyWorkflow): WorkflowOption => ({
  value: workflow.path,
  workflow
})

const options = computed<WorkflowOption[]>(() =>
  workflowStore.openWorkflows.map(workflowToOption)
)
const selectedWorkflow = computed<WorkflowOption | null>(() =>
  workflowStore.activeWorkflow
    ? workflowToOption(workflowStore.activeWorkflow as ComfyWorkflow)
    : null
)

const onWorkflowChange = async (option: WorkflowOption) => {
  // Prevent unselecting the current workflow
  if (!option) {
    return
  }
  // Prevent reloading the current workflow
  if (selectedWorkflow.value?.value === option.value) {
    return
  }

  await workflowService.openWorkflow(option.workflow)
}

const closeWorkflows = async (options: WorkflowOption[]) => {
  for (const opt of options) {
    if (
      !(await workflowService.closeWorkflow(opt.workflow, {
        warnIfUnsaved: !workspaceStore.shiftDown
      }))
    ) {
      // User clicked cancel
      break
    }
  }
}

const onCloseWorkflow = async (option: WorkflowOption) => {
  await closeWorkflows([option])
}

// Horizontal scroll on wheel
const handleWheel = (event: WheelEvent) => {
  const scrollElement = event.currentTarget as HTMLElement
  const scrollAmount = event.deltaX || event.deltaY
  scrollElement.scroll({
    left: scrollElement.scrollLeft + scrollAmount
  })
}

const scrollContent = computed(
  () =>
    (containerRef.value?.querySelector(
      '.p-scrollpanel-content'
    ) as HTMLElement | null) ?? null
)

const scroll = (direction: number) => {
  const el = scrollContent.value
  if (!el) return
  el.scrollBy({ left: direction * 20 })
}

const ensureActiveTabVisible = async (
  options: { waitForDom?: boolean } = {}
) => {
  if (!selectedWorkflow.value) return

  if (options.waitForDom !== false) {
    await nextTick()
  }

  const containerElement = containerRef.value
  if (!containerElement) return

  const activeTabElement = containerElement.querySelector(
    '.p-togglebutton-checked'
  )
  if (!activeTabElement) return

  activeTabElement.scrollIntoView({ block: 'nearest', inline: 'nearest' })
}

// Scroll to active offscreen tab when opened
watch(
  () => workflowStore.activeWorkflow,
  () => {
    void ensureActiveTabVisible()
  },
  { immediate: true }
)

let overflowObserver: ReturnType<typeof useOverflowObserver> | null = null
let stopArrivedWatch: WatchStopHandle | null = null
let stopOverflowWatch: WatchStopHandle | null = null

watch(
  scrollContent,
  (el, _prev, onCleanup) => {
    stopArrivedWatch?.()
    stopOverflowWatch?.()
    overflowObserver?.dispose()

    if (!el) return

    const scrollState = useScroll(el)

    stopArrivedWatch = watch(
      [
        () => scrollState.arrivedState.left,
        () => scrollState.arrivedState.right
      ],
      ([atLeft, atRight]) => {
        leftArrowEnabled.value = !atLeft
        rightArrowEnabled.value = !atRight
      },
      { immediate: true }
    )

    overflowObserver = useOverflowObserver(el)
    stopOverflowWatch = watch(
      overflowObserver.isOverflowing,
      (isOverflow) => {
        showOverflowArrows.value = isOverflow
        if (!isOverflow) return
        void nextTick(() => {
          // Force a new check after arrows are updated
          scrollState.measure()
          void ensureActiveTabVisible({ waitForDom: false })
        })
      },
      { immediate: true }
    )

    onCleanup(() => {
      stopArrivedWatch?.()
      stopOverflowWatch?.()
      overflowObserver?.dispose()
    })
  },
  { immediate: true }
)

onUpdated(() => {
  if (!overflowObserver?.disposed.value) {
    overflowObserver?.checkOverflow()
  }
})
</script>

<style scoped>
.workflow-tabs-container {
  background-color: var(--comfy-menu-bg);
}

:deep(.p-togglebutton) {
  position: relative;
  flex-shrink: 1;
  border: 0;
  border-right-style: solid;
  border-right-width: 1px;
  border-radius: 0;
  background-color: transparent;
  padding: 0;
  border-right-color: var(--border-color);
  min-width: 90px;
}

.overflow-arrow {
  border-radius: 0;
  padding-inline: calc(var(--spacing) * 2);
}

.overflow-arrow[disabled] {
  opacity: 0.25;
}

:deep(.p-togglebutton > .p-togglebutton-content) {
  max-width: 100%;
}

:deep(.workflow-tab) {
  max-width: 100%;
}

:deep(.p-togglebutton::before) {
  display: none;
}

:deep(.p-togglebutton:first-child) {
  border-left-style: solid;
  border-left-width: 1px;
  border-left-color: var(--border-color);
}

:deep(.p-togglebutton:not(:first-child)) {
  border-left-width: 0;
}

:deep(.p-togglebutton.p-togglebutton-checked) {
  height: 100%;
  border-bottom-style: solid;
  border-bottom-width: 1px;
  border-bottom-color: var(--p-button-text-primary-color);
}

:deep(.p-togglebutton:not(.p-togglebutton-checked)) {
  opacity: 0.75;
}

:deep(.p-togglebutton-checked) .close-button,
:deep(.p-togglebutton:hover) .close-button {
  visibility: visible;
}

:deep(.p-scrollpanel-content) {
  height: 100%;
}

:deep(.workflow-tabs) {
  display: flex;
}

/* Scrollbar half opacity to avoid blocking the active tab bottom border */
:deep(.p-scrollpanel:hover .p-scrollpanel-bar),
:deep(.p-scrollpanel:active .p-scrollpanel-bar) {
  opacity: 0.5;
}

:deep(.p-selectbutton) {
  height: 100%;
  border-radius: 0;
}

.workflow-tabs-container-desktop {
  max-width: env(titlebar-area-width, 100vw);
}

.window-actions-spacer {
  flex: auto;
  /* If we are using custom titlebar, then we need to add a gap for the user to drag the window */
  --window-actions-spacer-width: min(75px, env(titlebar-area-width, 0) * 9999);
  min-width: var(--window-actions-spacer-width);
}
</style>
