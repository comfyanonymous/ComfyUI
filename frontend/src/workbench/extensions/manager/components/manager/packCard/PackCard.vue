<template>
  <div
    :class="
      cn(
        'flex size-full flex-col overflow-hidden rounded-lg bg-modal-card-background transition-colors duration-200 cursor-pointer select-none',
        isSelected
          ? 'ring-3 ring-modal-card-border-highlighted'
          : 'hover:bg-modal-card-background-hovered',
        isDisabled && 'opacity-60'
      )
    "
  >
    <!-- Banner -->
    <div class="w-full rounded-t-lg">
      <PackBanner :node-pack="nodePack" />
    </div>

    <!-- Content -->
    <div class="flex flex-1 flex-col rounded-lg min-h-0">
      <div class="h-full w-full py-2 px-3">
        <div class="flex h-full w-full flex-col gap-y-1">
          <span
            class="truncate overflow-hidden text-xs font-bold text-ellipsis"
          >
            {{ nodePack.name }}
          </span>
          <p
            v-if="nodePack.description"
            class="my-0 mb-1 line-clamp-3 min-h-12 flex-1 overflow-hidden text-xs leading-4 font-medium break-words text-muted"
          >
            {{ nodePack.description }}
          </p>
          <div class="flex flex-col gap-y-2">
            <div class="flex flex-1 items-center gap-2">
              <div v-if="nodesLabel" class="p-2 pl-0 text-xs">
                {{ nodesLabel }}
              </div>
              <PackVersionBadge
                :node-pack="nodePack"
                :is-selected="isSelected"
                :fill="false"
                :class="isInstalling ? 'pointer-events-none' : ''"
              />
              <div
                v-if="formattedLatestVersionDate"
                class="flex items-center justify-center gap-1 px-2 py-1 text-xs font-medium text-muted"
              >
                {{ formattedLatestVersionDate }}
              </div>
            </div>
            <div class="flex">
              <span
                v-if="publisherName"
                class="max-w-40 truncate text-xs leading-3 font-medium text-muted"
              >
                {{ publisherName }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <div class="border-t border-border-default">
      <PackCardFooter :node-pack="nodePack" :is-installing="isInstalling" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, provide } from 'vue'
import { useI18n } from 'vue-i18n'

import { cn } from '@/utils/tailwindUtil'
import PackVersionBadge from '@/workbench/extensions/manager/components/manager/PackVersionBadge.vue'
import PackBanner from '@/workbench/extensions/manager/components/manager/packBanner/PackBanner.vue'
import PackCardFooter from '@/workbench/extensions/manager/components/manager/packCard/PackCardFooter.vue'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import {
  IsInstallingKey,
  isMergedNodePack
} from '@/workbench/extensions/manager/types/comfyManagerTypes'
import type {
  MergedNodePack,
  RegistryPack
} from '@/workbench/extensions/manager/types/comfyManagerTypes'

const { nodePack, isSelected = false } = defineProps<{
  nodePack: MergedNodePack | RegistryPack
  isSelected?: boolean
}>()

const { d, t } = useI18n()

const { isPackInstalled, isPackEnabled, isPackInstalling } =
  useComfyManagerStore()

const isInstalling = computed(() => isPackInstalling(nodePack?.id))
provide(IsInstallingKey, isInstalling)

const isInstalled = computed(() => isPackInstalled(nodePack?.id))
const isDisabled = computed(
  () => isInstalled.value && !isPackEnabled(nodePack?.id)
)

const nodesCount = computed(() =>
  isMergedNodePack(nodePack) ? nodePack.comfy_nodes?.length : undefined
)
const nodesLabel = computed(() =>
  nodesCount.value ? t('g.nodesCount', nodesCount.value) : ''
)
const publisherName = computed(() => {
  if (!nodePack) return null

  const { publisher, author } = nodePack
  return publisher?.name ?? publisher?.id ?? author
})

const formattedLatestVersionDate = computed(() => {
  if (!nodePack.latest_version?.createdAt) return null

  return d(new Date(nodePack.latest_version.createdAt), {
    dateStyle: 'medium'
  })
})
</script>
