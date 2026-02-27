<template>
  <div
    class="flex w-full max-w-[490px] flex-col border-t border-border-default"
  >
    <div class="flex h-full w-full flex-col gap-4 p-4">
      <p class="m-0 text-sm leading-5 text-muted-foreground">
        {{ $t('missingModelsDialog.description') }}
      </p>

      <div
        class="flex max-h-[300px] flex-col overflow-y-auto rounded-lg bg-secondary-background scrollbar-custom"
      >
        <div
          v-for="model in processedModels"
          :key="model.name"
          class="flex items-center justify-between px-3 py-2"
        >
          <div class="flex items-center gap-2 overflow-hidden">
            <span
              class="min-w-0 truncate text-sm text-foreground"
              :title="model.name"
            >
              {{ model.name }}
            </span>
            <span
              class="inline-flex h-4 shrink-0 items-center rounded-full bg-muted-foreground/20 px-1.5 text-xxxs font-semibold uppercase text-muted-foreground"
            >
              {{ model.badgeLabel }}
            </span>
          </div>
          <div class="flex shrink-0 items-center gap-2">
            <span
              v-if="model.isDownloadable && fileSizes.get(model.url)"
              class="text-xs text-muted-foreground"
            >
              {{ formatSize(fileSizes.get(model.url)) }}
            </span>
            <Button
              v-if="model.isDownloadable"
              variant="textonly"
              size="icon"
              :title="model.url"
              :aria-label="$t('g.download')"
              @click="downloadModel(model, paths)"
            >
              <i class="icon-[lucide--download] size-4" />
            </Button>
            <Button
              v-else
              variant="textonly"
              size="icon"
              :title="model.url"
              :aria-label="$t('g.copyURL')"
              @click="void copyToClipboard(model.url)"
            >
              <i class="icon-[lucide--copy] size-4" />
            </Button>
          </div>
        </div>

        <div
          v-if="totalDownloadSize > 0"
          class="sticky bottom-0 flex items-center justify-between border-t border-border-default bg-secondary-background px-3 py-2"
        >
          <span class="text-xs font-medium text-muted-foreground">
            {{ $t('missingModelsDialog.totalSize') }}
          </span>
          <span class="text-xs text-muted-foreground">
            {{ formatSize(totalDownloadSize) }}
          </span>
        </div>
      </div>

      <p
        class="m-0 text-xs leading-5 text-muted-foreground whitespace-pre-line"
      >
        {{ $t('missingModelsDialog.footerDescription') }}
      </p>

      <div
        v-if="hasCustomModels"
        class="flex gap-3 rounded-lg border border-warning-background bg-warning-background/10 p-3"
      >
        <i
          class="icon-[lucide--triangle-alert] mt-0.5 h-4 w-4 shrink-0 text-warning-background"
        />
        <div class="flex flex-col gap-1">
          <p
            class="m-0 text-xs font-semibold leading-5 text-warning-background"
          >
            {{ $t('missingModelsDialog.customModelsWarning') }}
          </p>
          <p class="m-0 text-xs leading-5 text-warning-background">
            {{ $t('missingModelsDialog.customModelsInstruction') }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { formatSize } from '@/utils/formatUtil'

import type { ModelWithUrl } from './missingModelsUtils'
import {
  downloadModel,
  getBadgeLabel,
  hasValidDirectory,
  isModelDownloadable
} from './missingModelsUtils'

const { missingModels, paths } = defineProps<{
  missingModels: ModelWithUrl[]
  paths: Record<string, string[]>
}>()

interface ProcessedModel {
  name: string
  url: string
  directory: string
  badgeLabel: string
  isDownloadable: boolean
}

const processedModels = computed<ProcessedModel[]>(() =>
  missingModels.map((model) => ({
    name: model.name,
    url: model.url,
    directory: model.directory,
    badgeLabel: getBadgeLabel(model.directory),
    isDownloadable:
      hasValidDirectory(model, paths) && isModelDownloadable(model)
  }))
)

const hasCustomModels = computed(() =>
  processedModels.value.some((m) => !m.isDownloadable)
)

const fileSizes = reactive(new Map<string, number>())

const totalDownloadSize = computed(() =>
  processedModels.value
    .filter((model) => model.isDownloadable)
    .reduce((total, model) => total + (fileSizes.get(model.url) ?? 0), 0)
)

onMounted(async () => {
  const downloadableUrls = processedModels.value
    .filter((m) => m.isDownloadable)
    .map((m) => m.url)

  await Promise.allSettled(
    downloadableUrls.map(async (url) => {
      try {
        const response = await fetch(url, { method: 'HEAD' })
        if (!response.ok) return
        const size = response.headers.get('content-length')
        if (size) fileSizes.set(url, parseInt(size, 10))
      } catch {
        // Silently skip size fetch failures
      }
    })
  )
})

const { copyToClipboard } = useCopyToClipboard()
</script>
