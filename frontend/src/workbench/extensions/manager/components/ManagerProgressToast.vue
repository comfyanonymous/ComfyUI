<script setup lang="ts">
import { useScroll, whenever } from '@vueuse/core'
import Panel from 'primevue/panel'
import TabMenu from 'primevue/tabmenu'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import DotSpinner from '@/components/common/DotSpinner.vue'
import HoneyToast from '@/components/honeyToast/HoneyToast.vue'
import Button from '@/components/ui/button/Button.vue'
import { useApplyChanges } from '@/workbench/extensions/manager/composables/useApplyChanges'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

const { t } = useI18n()
const comfyManagerStore = useComfyManagerStore()
const { isRestarting, isRestartCompleted, applyChanges } = useApplyChanges()

const isExpanded = ref(false)
const activeTabIndex = ref(0)

const tabs = computed(() => [
  { label: t('manager.installationQueue') },
  {
    label: t('manager.failed', {
      count: comfyManagerStore.failedTasksIds.length
    })
  }
])

const focusedLogs = computed(() => {
  if (activeTabIndex.value === 0) {
    return comfyManagerStore.succeededTasksLogs
  }
  return comfyManagerStore.failedTasksLogs
})

const visible = computed(() => comfyManagerStore.taskLogs.length > 0)

const isInProgress = computed(
  () => comfyManagerStore.isProcessingTasks || isRestarting.value
)

const isTaskInProgress = (index: number) => {
  const log = focusedLogs.value[index]
  if (!log) return false

  const taskQueue = comfyManagerStore.taskQueue
  if (!taskQueue) return false

  const allQueueTasks = [
    ...(taskQueue.running_queue || []),
    ...(taskQueue.pending_queue || [])
  ]

  return allQueueTasks.some((task) => task.ui_id === log.taskId)
}

const completedTasksCount = computed(() => {
  return (
    comfyManagerStore.succeededTasksIds.length +
    comfyManagerStore.failedTasksIds.length
  )
})

const totalTasksCount = computed(() => {
  const completedTasks = Object.keys(comfyManagerStore.taskHistory).length
  const taskQueue = comfyManagerStore.taskQueue
  const queuedTasks = taskQueue
    ? (taskQueue.running_queue?.length || 0) +
      (taskQueue.pending_queue?.length || 0)
    : 0
  return completedTasks + queuedTasks
})

const currentTaskName = computed(() => {
  if (isRestarting.value) {
    return t('manager.restartingBackend')
  }
  if (isRestartCompleted.value) {
    return t('manager.extensionsSuccessfullyInstalled')
  }
  if (!comfyManagerStore.taskLogs.length)
    return t('manager.installingDependencies')
  const task = comfyManagerStore.taskLogs.at(-1)
  return task?.taskName ?? t('manager.installingDependencies')
})

const collapsedPanels = ref<Record<number, boolean>>({})
function togglePanel(index: number) {
  collapsedPanels.value[index] = !collapsedPanels.value[index]
}

const sectionsContainerRef = ref<HTMLElement | null>(null)
const { y: scrollY } = useScroll(sectionsContainerRef, {
  eventListenerOptions: { passive: true }
})

const lastPanelRef = ref<HTMLElement | null>(null)
const isUserScrolling = ref(false)
const lastPanelLogs = computed(() => focusedLogs.value?.at(-1)?.logs)

function isAtBottom(el: HTMLElement | null) {
  if (!el) return false
  const threshold = 20
  return Math.abs(el.scrollHeight - el.scrollTop - el.clientHeight) < threshold
}

function scrollLastPanelToBottom() {
  if (!lastPanelRef.value || isUserScrolling.value) return
  lastPanelRef.value.scrollTop = lastPanelRef.value.scrollHeight
}

function scrollContentToBottom() {
  scrollY.value = sectionsContainerRef.value?.scrollHeight ?? 0
}

function resetUserScrolling() {
  isUserScrolling.value = false
}

function handleScroll(e: Event) {
  const target = e.target as HTMLElement
  if (target !== lastPanelRef.value) return
  isUserScrolling.value = !isAtBottom(target)
}

function onLogsAdded() {
  if (isUserScrolling.value) return
  scrollLastPanelToBottom()
}

whenever(lastPanelLogs, onLogsAdded, { flush: 'post', deep: true })
whenever(() => isExpanded.value, scrollContentToBottom)
whenever(() => !isExpanded.value, resetUserScrolling)

function closeToast() {
  comfyManagerStore.resetTaskState()
  isExpanded.value = false
}

async function handleRestart() {
  try {
    await applyChanges(closeToast)
  } catch (err) {
    console.error('[ManagerProgressToast] Restart failed:', err)
  }
}

onMounted(() => {
  scrollContentToBottom()
})

onBeforeUnmount(() => {
  isExpanded.value = false
})
</script>

<template>
  <HoneyToast v-model:expanded="isExpanded" :visible>
    <template #default>
      <div v-if="isExpanded" class="flex items-center px-4 py-2">
        <TabMenu
          v-model:active-index="activeTabIndex"
          :model="tabs"
          class="w-full border-none"
          :pt="{
            menu: { class: 'border-none' },
            menuitem: { class: 'font-medium' },
            action: { class: 'px-4 py-2' }
          }"
        />
      </div>

      <div
        ref="sectionsContainerRef"
        class="scroll-container max-h-[450px] overflow-y-auto px-6 py-4"
        :style="{
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(156, 163, 175, 0.5) transparent'
        }"
      >
        <div v-for="(log, index) in focusedLogs" :key="index">
          <Panel
            :expanded="collapsedPanels[index] === true"
            toggleable
            class="shadow-elevation-1 mt-2 rounded-lg"
          >
            <template #header>
              <div class="flex w-full items-center justify-between py-2">
                <div class="flex flex-col text-sm leading-normal font-medium">
                  <span>{{ log.taskName }}</span>
                  <span class="text-muted">
                    {{
                      isTaskInProgress(index)
                        ? t('g.inProgress')
                        : t('g.completedWithCheckmark')
                    }}
                  </span>
                </div>
              </div>
            </template>
            <template #toggleicon>
              <Button
                variant="textonly"
                class="text-neutral-300"
                @click="togglePanel(index)"
              >
                <i
                  :class="
                    collapsedPanels[index]
                      ? 'pi pi-chevron-right'
                      : 'pi pi-chevron-down'
                  "
                />
              </Button>
            </template>
            <div
              :ref="
                index === focusedLogs.length - 1
                  ? (el) => (lastPanelRef = el as HTMLElement)
                  : undefined
              "
              class="h-64 overflow-y-auto rounded-lg bg-black"
              :class="{
                'h-64': index !== focusedLogs.length - 1,
                grow: index === focusedLogs.length - 1
              }"
              @scroll="handleScroll"
            >
              <div class="h-full">
                <div
                  v-for="(logLine, logIndex) in log.logs"
                  :key="logIndex"
                  class="text-muted"
                >
                  <pre class="break-words whitespace-pre-wrap">{{
                    logLine
                  }}</pre>
                </div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </template>

    <template #footer="{ toggle }">
      <div class="flex w-full items-center justify-between px-6 py-2 shadow-lg">
        <div class="flex items-center text-base leading-none">
          <div class="flex items-center">
            <template v-if="isInProgress">
              <DotSpinner duration="1s" class="mr-2" />
              <span>{{ currentTaskName }}</span>
            </template>
            <template v-else-if="isRestartCompleted">
              <span class="mr-2">🎉</span>
              <span>{{ currentTaskName }}</span>
            </template>
            <template v-else>
              <span class="mr-2">✅</span>
              <span>{{ t('manager.restartToApplyChanges') }}</span>
            </template>
          </div>
        </div>
        <div class="flex items-center gap-4">
          <span v-if="isInProgress" class="text-sm text-muted-foreground">
            {{ completedTasksCount }} {{ t('g.progressCountOf') }}
            {{ totalTasksCount }}
          </span>
          <div class="flex items-center">
            <Button
              v-if="!isInProgress && !isRestartCompleted"
              variant="secondary"
              class="mr-4 rounded-full border-2 border-base-foreground px-3 text-base-foreground hover:bg-secondary-background-hover"
              @click="handleRestart"
            >
              {{ t('manager.applyChanges') }}
            </Button>
            <Button
              v-else-if="!isRestartCompleted"
              variant="muted-textonly"
              size="sm"
              class="rounded-full font-bold"
              :aria-label="
                t(isExpanded ? 'contextMenu.Collapse' : 'contextMenu.Expand')
              "
              @click.stop="toggle"
            >
              <i
                :class="isExpanded ? 'pi pi-chevron-up' : 'pi pi-chevron-down'"
              />
            </Button>
            <Button
              variant="muted-textonly"
              size="sm"
              class="rounded-full font-bold"
              :aria-label="t('g.close')"
              @click.stop="closeToast"
            >
              <i class="pi pi-times" />
            </Button>
          </div>
        </div>
      </div>
    </template>
  </HoneyToast>
</template>
