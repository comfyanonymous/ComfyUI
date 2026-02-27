<template>
  <BaseViewTemplate dark>
    <div
      class="min-w-full min-h-full font-sans w-screen h-screen grid justify-around text-neutral-300 bg-neutral-900 dark-theme overflow-y-auto"
    >
      <div class="max-w-(--breakpoint-sm) w-screen m-8 relative">
        <!-- Header -->
        <h1 class="backspan pi-wrench text-4xl font-bold">
          {{ t('maintenance.title') }}
        </h1>

        <!-- Toolbar -->
        <div class="w-full flex flex-wrap gap-4 items-center">
          <span class="grow">
            {{ t('maintenance.status') }}:
            <StatusTag :refreshing="isRefreshing" :error="anyErrors" />
          </span>
          <div class="flex gap-4 items-center">
            <SelectButton
              v-model="displayAsList"
              :options="[PrimeIcons.LIST, PrimeIcons.TH_LARGE]"
              :allow-empty="false"
            >
              <template #option="opts">
                <i :class="opts.option" />
              </template>
            </SelectButton>
            <SelectButton
              v-model="filter"
              :options="filterOptions"
              :allow-empty="false"
              option-label="value"
              data-key="value"
              area-labelledby="custom"
              @change="clearResolved"
            >
              <template #option="opts">
                <i :class="opts.option.icon" />
                <span class="max-sm:hidden">{{ opts.option.value }}</span>
              </template>
            </SelectButton>
            <RefreshButton
              v-model="isRefreshing"
              severity="secondary"
              @refresh="refreshDesktopTasks"
            />
          </div>
        </div>

        <!-- Unsafe migration warning -->
        <div v-if="taskStore.unsafeBasePath" class="my-4">
          <p class="flex items-start gap-3 text-neutral-300">
            <Tag
              icon="pi pi-exclamation-triangle"
              severity="warn"
              :value="t('icon.exclamation-triangle')"
            />
            <span>
              <strong class="block mb-1">
                {{ t('maintenance.unsafeMigration.title') }}
              </strong>
              <span class="block mb-1">
                {{ unsafeReasonText }}
              </span>
              <span class="block text-sm text-neutral-400">
                {{ t('maintenance.unsafeMigration.action') }}
              </span>
            </span>
          </p>
        </div>

        <!-- Tasks -->
        <TaskListPanel
          class="border-neutral-700 border-solid border-x-0 border-y"
          :filter
          :display-as-list
        />

        <!-- Actions -->
        <div class="flex justify-between gap-4 flex-row">
          <Button
            :label="t('maintenance.consoleLogs')"
            icon="pi pi-desktop"
            icon-pos="left"
            severity="secondary"
            @click="toggleConsoleDrawer"
          />
          <Button
            :label="t('g.continue')"
            icon="pi pi-arrow-right"
            icon-pos="left"
            :severity="anyErrors ? 'secondary' : 'primary'"
            :loading="isRefreshing"
            @click="() => completeValidation()"
          />
        </div>
      </div>

      <TerminalOutputDrawer
        v-model="terminalVisible"
        :header="t('g.terminal')"
        :default-message="t('maintenance.terminalDefaultMessage')"
      />
      <Toast />
    </div>
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import { PrimeIcons } from '@primevue/core/api'
import Button from 'primevue/button'
import SelectButton from 'primevue/selectbutton'
import Tag from 'primevue/tag'
import Toast from 'primevue/toast'
import { useToast } from 'primevue/usetoast'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import RefreshButton from '@/components/common/RefreshButton.vue'
import StatusTag from '@/components/maintenance/StatusTag.vue'
import TaskListPanel from '@/components/maintenance/TaskListPanel.vue'
import TerminalOutputDrawer from '@/components/maintenance/TerminalOutputDrawer.vue'
import { useMaintenanceTaskStore } from '@/stores/maintenanceTaskStore'
import type { MaintenanceFilter } from '@/types/desktop/maintenanceTypes'
import { electronAPI } from '@/utils/envUtil'
import { useMinLoadingDurationRef } from '@/utils/refUtil'

import BaseViewTemplate from './templates/BaseViewTemplate.vue'

const electron = electronAPI()
const toast = useToast()
const { t } = useI18n()
const taskStore = useMaintenanceTaskStore()
const { clearResolved, processUpdate, refreshDesktopTasks } = taskStore

const terminalVisible = ref(false)

// Use a minimum run time to ensure tasks "feel" like they have run
const reactiveIsRefreshing = computed(() => taskStore.isRefreshing)
/** `true` when waiting on tasks to complete. */
const isRefreshing = useMinLoadingDurationRef(reactiveIsRefreshing, 250)

/** True if any tasks are in an error state. */
const anyErrors = computed(() => taskStore.anyErrors)

/** Whether to display tasks as a list or cards. */
const displayAsList = ref(PrimeIcons.TH_LARGE)

const errorFilter = computed(() =>
  taskStore.tasks.filter((x) => {
    const { state, resolved } = taskStore.getRunner(x)
    return state === 'error' || resolved
  })
)

const filterOptions = ref([
  { icon: PrimeIcons.FILTER_FILL, value: 'All', tasks: taskStore.tasks },
  { icon: PrimeIcons.EXCLAMATION_TRIANGLE, value: 'Errors', tasks: errorFilter }
])

/** Filter binding; can be set to show all tasks, or only errors. */
const filter = ref<MaintenanceFilter>(filterOptions.value[0])

const unsafeReasonText = computed(() => {
  const reason = taskStore.unsafeBasePathReason
  if (!reason) {
    return t('maintenance.unsafeMigration.generic')
  }

  if (reason === 'appInstallDir') {
    return t('maintenance.unsafeMigration.appInstallDir')
  }

  if (reason === 'updaterCache') {
    return t('maintenance.unsafeMigration.updaterCache')
  }

  if (reason === 'oneDrive') {
    return t('maintenance.unsafeMigration.oneDrive')
  }

  return t('maintenance.unsafeMigration.generic')
})

/** If valid, leave the validation window. */
const completeValidation = async () => {
  const isValid = await electron.Validation.complete()
  if (!isValid) {
    toast.add({
      severity: 'error',
      summary: t('g.error'),
      detail: t('maintenance.error.cannotContinue'),
      life: 5_000
    })
  }
}

const toggleConsoleDrawer = () => {
  terminalVisible.value = !terminalVisible.value
}

// Show terminal when in use
watch(
  () => taskStore.isRunningTerminalCommand,
  (value) => {
    terminalVisible.value = value
  }
)

onMounted(async () => {
  electron.Validation.onUpdate(processUpdate)

  const update = await electron.Validation.getStatus()
  if (Object.values(update).some((x) => x === 'error')) {
    filter.value = filterOptions.value[1]
  }
  processUpdate(update)
})

onUnmounted(() => electron.Validation.dispose())
</script>

<style scoped>
:deep(.p-tag) {
  --p-tag-gap: 0.375rem;
}

.backspan::before {
  position: absolute;
  margin: 0;
  color: var(--muted-foreground);
  font-family: 'primeicons', sans-serif;
  top: -2rem;
  right: -2rem;
  speak: none;
  font-style: normal;
  font-weight: 400;
  font-variant: normal;
  text-transform: none;
  line-height: 1;
  display: inline-block;
  -webkit-font-smoothing: antialiased;
  opacity: 0.02;
  font-size: min(14rem, 90vw);
  z-index: 0;
}
</style>
