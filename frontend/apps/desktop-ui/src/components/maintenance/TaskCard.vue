<template>
  <div
    :class="
      cn(
        'task-div group/task-card relative grid min-h-52 max-w-48',
        isLoading && 'opacity-75'
      )
    "
  >
    <Card
      :class="
        cn(
          'relative h-full max-w-48 overflow-hidden',
          runner.state !== 'error' && 'opacity-65'
        )
      "
      :pt="cardPt"
      v-bind="(({ onClick, ...rest }) => rest)($attrs)"
    >
      <template #header>
        <i
          v-if="runner.state === 'error'"
          class="pi pi-exclamation-triangle absolute top-0 -right-14 m-2 text-red-500 opacity-15"
          style="font-size: 10rem"
        />
        <img
          v-if="task.headerImg"
          :src="task.headerImg"
          class="h-full w-full object-contain px-4 pt-4 opacity-25"
        />
      </template>
      <template #title>
        {{ task.name }}
      </template>
      <template #content>
        {{ description }}
      </template>
      <template #footer>
        <div class="mt-1 flex gap-4">
          <Button
            :icon="task.button?.icon"
            :label="task.button?.text"
            class="w-full"
            raised
            icon-pos="right"
            :loading="isExecuting"
            @click="(event) => $emit('execute', event)"
          />
        </div>
      </template>
    </Card>

    <i
      v-if="!isLoading && runner.state === 'OK'"
      class="pi pi-check pointer-events-none absolute -right-4 -bottom-4 col-span-full row-span-full z-10 text-[4rem] text-green-500 opacity-100 transition-opacity group-hover/task-card:opacity-20 [text-shadow:0.25rem_0_0.5rem_black]"
    />
  </div>
</template>

<script setup lang="ts">
import Button from 'primevue/button'
import Card from 'primevue/card'
import { computed } from 'vue'

import { useMaintenanceTaskStore } from '@/stores/maintenanceTaskStore'
import type { MaintenanceTask } from '@/types/desktop/maintenanceTypes'
import { cn } from '@/utils/tailwindUtil'
import { useMinLoadingDurationRef } from '@/utils/refUtil'

const taskStore = useMaintenanceTaskStore()
const runner = computed(() => taskStore.getRunner(props.task))

// Properties
const props = defineProps<{
  task: MaintenanceTask
}>()

// Events
defineEmits<{
  execute: [event: MouseEvent]
}>()

// Bindings
const description = computed(() =>
  runner.value.state === 'error'
    ? (props.task.errorDescription ?? props.task.shortDescription)
    : props.task.shortDescription
)

// Use a minimum run time to ensure tasks "feel" like they have run
const reactiveLoading = computed(() => !!runner.value.refreshing)
const reactiveExecuting = computed(() => !!runner.value.executing)

const isLoading = useMinLoadingDurationRef(reactiveLoading, 250)
const isExecuting = useMinLoadingDurationRef(reactiveExecuting, 250)

const cardPt = {
  header: { class: 'z-0' },
  body: { class: 'z-[1] grow justify-between' }
}
</script>
