<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import HoneyToast from '@/components/honeyToast/HoneyToast.vue'
import Button from '@/components/ui/button/Button.vue'
import type { AssetExport } from '@/stores/assetExportStore'
import { useAssetExportStore } from '@/stores/assetExportStore'
import { cn } from '@/utils/tailwindUtil'

const { t } = useI18n()
const assetExportStore = useAssetExportStore()

const visible = computed(() => assetExportStore.hasExports)
const isExpanded = ref(false)

const exportJobs = computed(() => assetExportStore.exportList)
const failedJobs = computed(() =>
  assetExportStore.finishedExports.filter((e) => e.status === 'failed')
)

const isInProgress = computed(() => assetExportStore.hasActiveExports)
const currentJobName = computed(() => {
  const activeJob = exportJobs.value.find((job) => job.status === 'running')
  return activeJob?.exportName || t('exportToast.preparingExport')
})

function jobDisplayName(job: AssetExport): string {
  if (job.status === 'failed') return job.error || t('exportToast.exportError')
  return job.exportName || t('exportToast.preparingExport')
}

const completedCount = computed(() => assetExportStore.finishedExports.length)
const totalCount = computed(() => exportJobs.value.length)

const footerLabel = computed(() => {
  if (isInProgress.value) return currentJobName.value
  if (failedJobs.value.length > 0)
    return t('exportToast.exportFailed', { count: failedJobs.value.length })
  return t('exportToast.allExportsCompleted')
})

const footerIconClass = computed(() => {
  if (isInProgress.value)
    return 'icon-[lucide--loader-circle] animate-spin text-muted-foreground'
  if (failedJobs.value.length > 0)
    return 'icon-[lucide--circle-alert] text-destructive-background'
  return 'icon-[lucide--check-circle] text-jade-600'
})

const tooltipConfig = computed(() => ({
  value: footerLabel.value,
  disabled: isExpanded.value,
  pt: { root: { class: 'z-10000!' } }
}))

function progressPercent(job: AssetExport): number {
  return Math.round(job.progress * 100)
}

function closeDialog() {
  assetExportStore.clearFinishedExports()
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
          {{ t('exportToast.exportingAssets') }}
        </h3>
      </div>

      <div class="relative max-h-75 overflow-y-auto px-4 py-4">
        <div class="flex flex-col gap-2">
          <div
            v-for="job in exportJobs"
            :key="job.taskId"
            :class="
              cn(
                'flex items-center justify-between rounded-lg bg-modal-card-background px-4 py-3',
                job.status === 'completed' && 'opacity-50'
              )
            "
          >
            <div class="min-w-0 flex-1">
              <span
                :class="
                  cn(
                    'block truncate text-sm',
                    job.status === 'failed'
                      ? 'text-destructive-background'
                      : 'text-base-foreground'
                  )
                "
              >
                {{ jobDisplayName(job) }}
              </span>
              <span
                v-if="job.assetsTotal > 0"
                class="text-xs text-muted-foreground"
              >
                {{ job.assetsAttempted }}/{{ job.assetsTotal }}
              </span>
            </div>

            <div class="flex items-center gap-2">
              <template v-if="job.status === 'failed'">
                <i
                  class="icon-[lucide--circle-alert] size-4 text-destructive-background"
                />
              </template>
              <template
                v-else-if="job.status === 'completed' && job.downloadError"
              >
                <span
                  class="text-xs text-destructive-background truncate max-w-32"
                >
                  {{ job.downloadError }}
                </span>
                <Button
                  variant="muted-textonly"
                  size="icon"
                  :aria-label="t('exportToast.retryDownload')"
                  @click.stop="assetExportStore.triggerDownload(job, true)"
                >
                  <i
                    class="icon-[lucide--rotate-ccw] size-4 text-destructive-background"
                  />
                </Button>
              </template>
              <template v-else-if="job.status === 'completed'">
                <Button
                  variant="muted-textonly"
                  size="icon"
                  :aria-label="t('exportToast.downloadExport')"
                  @click.stop="assetExportStore.triggerDownload(job, true)"
                >
                  <i
                    class="icon-[lucide--download] size-4 text-success-background"
                  />
                </Button>
              </template>
              <template v-else-if="job.status === 'running'">
                <i
                  class="icon-[lucide--loader-circle] size-4 animate-spin text-base-foreground"
                />
                <span class="text-xs text-base-foreground">
                  {{ progressPercent(job) }}%
                </span>
              </template>
              <template v-else>
                <span class="text-xs text-muted-foreground">
                  {{ t('progressToast.pending') }}
                </span>
              </template>
            </div>
          </div>
        </div>

        <div
          v-if="exportJobs.length === 0"
          class="flex flex-col items-center justify-center py-6 text-center"
        >
          <span class="text-sm text-muted-foreground">
            {{
              t('exportToast.noExportsInQueue', {
                filter: t('progressToast.filter.all')
              })
            }}
          </span>
        </div>
      </div>
    </template>

    <template #footer="{ toggle }">
      <div
        class="flex flex-1 min-w-0 h-12 items-center justify-between gap-2 border-t border-border-default px-4"
      >
        <div class="flex min-w-0 flex-1 items-center gap-2 text-sm">
          <i
            v-tooltip.top="tooltipConfig"
            :class="cn('size-4 shrink-0', footerIconClass)"
          />
          <span
            :class="
              cn(
                'truncate font-bold text-base-foreground transition-all duration-300 overflow-hidden',
                isExpanded ? 'min-w-0 flex-1' : 'w-0'
              )
            "
          >
            {{ footerLabel }}
          </span>
        </div>

        <div class="flex items-center gap-2">
          <span
            v-if="isInProgress"
            :class="
              cn(
                'text-sm text-muted-foreground transition-all duration-300 overflow-hidden',
                isExpanded ? 'whitespace-nowrap' : 'w-0'
              )
            "
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
