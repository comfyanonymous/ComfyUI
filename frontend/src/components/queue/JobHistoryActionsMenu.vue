<template>
  <div class="flex items-center gap-1">
    <Popover :show-arrow="false">
      <template #button>
        <Button
          v-tooltip.top="moreTooltipConfig"
          variant="textonly"
          size="icon"
          :aria-label="t('sideToolbar.queueProgressOverlay.moreOptions')"
        >
          <i
            class="icon-[lucide--more-horizontal] block size-4 leading-none text-text-secondary"
          />
        </Button>
      </template>
      <template #default="{ close }">
        <div class="flex min-w-[14rem] flex-col items-stretch font-inter">
          <Button
            data-testid="docked-job-history-action"
            class="w-full justify-between text-sm font-light"
            variant="textonly"
            size="sm"
            @click="onToggleDockedJobHistory"
          >
            <span class="flex items-center gap-2">
              <i
                class="icon-[lucide--panel-left-close] size-4 text-text-secondary"
              />
              <span>{{
                t('sideToolbar.queueProgressOverlay.dockedJobHistory')
              }}</span>
            </span>
            <i
              v-if="isQueuePanelV2Enabled"
              class="icon-[lucide--check] size-4"
            />
          </Button>
          <!-- TODO: Bug in assets sidebar panel derives assets from history, so despite this not deleting the assets, it still effectively shows to the user as deleted -->
          <template v-if="showClearHistoryAction">
            <div class="my-1 border-t border-interface-stroke" />
            <Button
              data-testid="clear-history-action"
              class="h-auto min-h-0 w-full items-start justify-start whitespace-normal"
              variant="textonly"
              size="sm"
              @click="onClearHistoryFromMenu(close)"
            >
              <i
                class="icon-[lucide--trash-2] size-4 shrink-0 self-center text-destructive-background"
              />
              <span
                class="flex flex-col items-start break-words text-left leading-tight"
              >
                <span class="text-sm font-light">
                  {{ t('sideToolbar.queueProgressOverlay.clearHistory') }}
                </span>
                <span class="text-xs text-text-secondary font-light">
                  {{
                    t(
                      'sideToolbar.queueProgressOverlay.clearHistoryMenuAssetsNote'
                    )
                  }}
                </span>
              </span>
            </Button>
          </template>
        </div>
      </template>
    </Popover>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Popover from '@/components/ui/Popover.vue'
import Button from '@/components/ui/button/Button.vue'
import { buildTooltipConfig } from '@/composables/useTooltipConfig'
import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'

const emit = defineEmits<{
  (e: 'clearHistory'): void
}>()

const { t } = useI18n()
const settingStore = useSettingStore()
const sidebarTabStore = useSidebarTabStore()

const moreTooltipConfig = computed(() => buildTooltipConfig(t('g.more')))
const isQueuePanelV2Enabled = computed(() =>
  settingStore.get('Comfy.Queue.QPOV2')
)
const showClearHistoryAction = computed(() => !isCloud)

const onClearHistoryFromMenu = (close: () => void) => {
  close()
  emit('clearHistory')
}

const onToggleDockedJobHistory = async () => {
  if (isQueuePanelV2Enabled.value) {
    await settingStore.set('Comfy.Queue.QPOV2', false)
    await settingStore.set('Comfy.Queue.History.Expanded', true)
    return
  }

  await settingStore.set('Comfy.Queue.QPOV2', true)
  sidebarTabStore.activeSidebarTabId = 'job-history'
}
</script>
