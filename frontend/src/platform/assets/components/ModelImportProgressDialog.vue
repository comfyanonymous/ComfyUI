<script setup lang="ts">
import { whenever } from '@vueuse/core'
import Popover from 'primevue/popover'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import HoneyToast from '@/components/honeyToast/HoneyToast.vue'
import ProgressToastItem from '@/components/toast/ProgressToastItem.vue'
import Button from '@/components/ui/button/Button.vue'
import { useAssetDownloadStore } from '@/stores/assetDownloadStore'
import { cn } from '@/utils/tailwindUtil'

const { t } = useI18n()
const assetDownloadStore = useAssetDownloadStore()

const visible = computed(() => assetDownloadStore.hasDownloads)

const isExpanded = ref(false)
const activeFilter = ref<'all' | 'completed' | 'failed'>('all')
const filterPopoverRef = ref<InstanceType<typeof Popover> | null>(null)

whenever(
  () => !isExpanded.value,
  () => filterPopoverRef.value?.hide()
)

const filterOptions = [
  { value: 'all', label: 'all' },
  { value: 'completed', label: 'completed' },
  { value: 'failed', label: 'failed' }
] as const

function onFilterClick(event: Event) {
  filterPopoverRef.value?.toggle(event)
}

function setFilter(filter: typeof activeFilter.value) {
  activeFilter.value = filter
  filterPopoverRef.value?.hide()
}

const downloadJobs = computed(() => assetDownloadStore.downloadList)
const completedJobs = computed(() =>
  assetDownloadStore.finishedDownloads.filter((d) => d.status === 'completed')
)
const failedJobs = computed(() =>
  assetDownloadStore.finishedDownloads.filter((d) => d.status === 'failed')
)

const isInProgress = computed(() => assetDownloadStore.hasActiveDownloads)
const currentJobName = computed(() => {
  const activeJob = downloadJobs.value.find((job) => job.status === 'running')
  return activeJob?.assetName || t('progressToast.downloadingModel')
})

const completedCount = computed(
  () => completedJobs.value.length + failedJobs.value.length
)
const totalCount = computed(() => downloadJobs.value.length)

const filteredJobs = computed(() => {
  switch (activeFilter.value) {
    case 'completed':
      return completedJobs.value
    case 'failed':
      return failedJobs.value
    default:
      return downloadJobs.value
  }
})

const activeFilterLabel = computed(() => {
  const option = filterOptions.find((f) => f.value === activeFilter.value)
  return option
    ? t(`progressToast.filter.${option.label}`)
    : t('progressToast.filter.all')
})

function closeDialog() {
  assetDownloadStore.clearFinishedDownloads()
  isExpanded.value = false
}
</script>

<template>
  <HoneyToast v-model:expanded="isExpanded" :visible>
    <template #default>
      <div
        class="flex h-12 items-center justify-between border-b border-border-default px-4"
      >
        <h3 class="text-sm font-bold text-base-foreground">
          {{ t('progressToast.importingModels') }}
        </h3>
        <div class="flex items-center gap-2">
          <Button
            variant="secondary"
            size="md"
            class="gap-1.5 px-2"
            @click="onFilterClick"
          >
            <i class="icon-[lucide--list-filter] size-4" />
            <span>{{ activeFilterLabel }}</span>
            <i class="icon-[lucide--chevron-down] size-3" />
          </Button>
          <Popover
            ref="filterPopoverRef"
            :dismissable="true"
            :close-on-escape="true"
            unstyled
            :base-z-index="9999"
            :pt="{
              root: { class: 'absolute z-50' },
              content: {
                class:
                  'bg-transparent border-none p-0 pt-2 rounded-lg shadow-lg'
              }
            }"
          >
            <div
              class="flex min-w-30 flex-col items-stretch rounded-lg border border-interface-stroke bg-interface-panel-surface px-2 py-3"
            >
              <Button
                v-for="option in filterOptions"
                :key="option.value"
                variant="textonly"
                size="sm"
                :class="
                  cn(
                    'w-full justify-start bg-transparent',
                    activeFilter === option.value &&
                      'bg-secondary-background-selected'
                  )
                "
                @click="setFilter(option.value)"
              >
                {{ t(`progressToast.filter.${option.label}`) }}
              </Button>
            </div>
          </Popover>
        </div>
      </div>

      <div class="relative max-h-75 overflow-y-auto px-4 py-4">
        <div
          v-if="filteredJobs.length > 3"
          class="absolute right-1 top-4 h-12 w-1 rounded-full bg-muted-foreground"
        />

        <div class="flex flex-col gap-2">
          <ProgressToastItem
            v-for="job in filteredJobs"
            :key="job.taskId"
            :job="job"
          />
        </div>

        <div
          v-if="filteredJobs.length === 0"
          class="flex flex-col items-center justify-center py-6 text-center"
        >
          <span class="text-sm text-muted-foreground">
            {{
              t('progressToast.noImportsInQueue', {
                filter: activeFilterLabel
              })
            }}
          </span>
        </div>
      </div>
    </template>

    <template #footer="{ toggle }">
      <div
        class="flex h-12 items-center justify-between gap-2 border-t border-border-default px-4"
      >
        <div class="flex min-w-0 flex-1 items-center gap-2 text-sm">
          <template v-if="isInProgress">
            <i
              class="icon-[lucide--loader-circle] size-4 flex-shrink-0 animate-spin text-muted-foreground"
            />
            <span
              class="min-w-0 flex-1 truncate font-bold text-base-foreground"
            >
              {{ currentJobName }}
            </span>
          </template>
          <template v-else-if="failedJobs.length > 0">
            <i
              class="icon-[lucide--circle-alert] size-4 flex-shrink-0 text-destructive-background"
            />
            <span class="min-w-0 truncate font-bold text-base-foreground">
              {{
                t('progressToast.downloadsFailed', {
                  count: failedJobs.length
                })
              }}
            </span>
          </template>
          <template v-else>
            <i
              class="icon-[lucide--check-circle] size-4 flex-shrink-0 text-jade-600"
            />
            <span class="min-w-0 truncate font-bold text-base-foreground">
              {{ t('progressToast.allDownloadsCompleted') }}
            </span>
          </template>
        </div>

        <div class="flex flex-shrink-0 items-center gap-2">
          <span
            v-if="isInProgress"
            class="whitespace-nowrap text-sm text-muted-foreground"
          >
            {{
              t('progressToast.progressCount', {
                completed: completedCount,
                total: totalCount
              })
            }}
          </span>

          <div class="flex items-center">
            <Button
              variant="muted-textonly"
              size="icon"
              :aria-label="
                isExpanded ? t('contextMenu.Collapse') : t('contextMenu.Expand')
              "
              @click.stop="toggle"
            >
              <i
                :class="
                  cn(
                    'size-4',
                    isExpanded
                      ? 'icon-[lucide--chevron-down]'
                      : 'icon-[lucide--chevron-up]'
                  )
                "
              />
            </Button>

            <Button
              v-if="!isInProgress"
              variant="muted-textonly"
              size="icon"
              :aria-label="t('g.close')"
              @click.stop="closeDialog"
            >
              <i class="icon-[lucide--x] size-4" />
            </Button>
          </div>
        </div>
      </div>
    </template>
  </HoneyToast>
</template>
