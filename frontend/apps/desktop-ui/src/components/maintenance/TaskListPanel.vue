<template>
  <!-- Tasks -->
  <section class="my-4">
    <template v-if="filter.tasks.length === 0">
      <!-- Empty filter -->
      <Divider />
      <p class="text-neutral-400 w-full text-center">
        {{ $t('maintenance.allOk') }}
      </p>
    </template>

    <template v-else>
      <!-- Display: List -->
      <table
        v-if="displayAsList === PrimeIcons.LIST"
        class="w-full border-collapse border-hidden"
      >
        <TaskListItem
          v-for="task in filter.tasks"
          :key="task.id"
          :task
          @execute="(event) => confirmButton(event, task)"
        />
      </table>

      <!-- Display: Cards -->
      <template v-else>
        <div class="flex flex-wrap justify-evenly gap-8 pad-y my-4">
          <TaskCard
            v-for="task in filter.tasks"
            :key="task.id"
            :task
            @execute="(event) => confirmButton(event, task)"
          />
        </div>
      </template>
    </template>
    <ConfirmPopup />
  </section>
</template>

<script setup lang="ts">
import { PrimeIcons } from '@primevue/core/api'
import { useConfirm, useToast } from 'primevue'
import ConfirmPopup from 'primevue/confirmpopup'
import Divider from 'primevue/divider'

import { t } from '@/i18n'
import { useMaintenanceTaskStore } from '@/stores/maintenanceTaskStore'
import type {
  MaintenanceFilter,
  MaintenanceTask
} from '@/types/desktop/maintenanceTypes'

import TaskCard from './TaskCard.vue'
import TaskListItem from './TaskListItem.vue'

const toast = useToast()
const confirm = useConfirm()
const taskStore = useMaintenanceTaskStore()

// Properties
defineProps<{
  displayAsList: string
  filter: MaintenanceFilter
}>()

const executeTask = async (task: MaintenanceTask) => {
  let message: string | undefined

  try {
    // Success
    if ((await taskStore.execute(task)) === true) return

    message = t('maintenance.error.taskFailed')
  } catch (error) {
    message = (error as Error)?.message
  }

  toast.add({
    severity: 'error',
    summary: t('maintenance.error.toastTitle'),
    detail: message ?? t('maintenance.error.defaultDescription'),
    life: 10_000
  })
}

// Commands
const confirmButton = async (event: MouseEvent, task: MaintenanceTask) => {
  if (!task.requireConfirm) {
    await executeTask(task)
    return
  }

  confirm.require({
    target: event.currentTarget as HTMLElement,
    message: task.confirmText ?? t('maintenance.confirmTitle'),
    icon: 'pi pi-exclamation-circle',
    rejectProps: {
      label: t('g.cancel'),
      severity: 'secondary',
      outlined: true
    },
    acceptProps: {
      label: task.button?.text ?? t('g.save'),
      severity: task.severity ?? 'primary'
    },
    // TODO: Not awaited.
    accept: async () => {
      await executeTask(task)
    }
  })
}
</script>
