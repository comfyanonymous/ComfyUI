<template>
  <div
    data-component-id="AssetCard"
    :data-asset-id="asset.id"
    :aria-labelledby="titleId"
    :aria-describedby="descId"
    :tabindex="interactive ? 0 : -1"
    :class="
      cn(
        'select-none rounded-2xl overflow-hidden transition-all duration-200 bg-modal-card-background p-2 gap-2 flex flex-col h-full',
        interactive &&
          'group appearance-none bg-transparent m-0 outline-none text-left hover:bg-secondary-background focus:bg-secondary-background border-none focus:outline-solid outline-base-foreground outline-4',
        focused && 'bg-secondary-background outline-solid'
      )
    "
    @click.stop="interactive && $emit('focus', asset)"
    @focus="interactive && $emit('focus', asset)"
    @keydown.enter.self="interactive && $emit('select', asset)"
  >
    <div class="relative aspect-square w-full overflow-hidden rounded-xl">
      <div
        v-if="isLoading || error"
        class="flex size-full cursor-pointer items-center justify-center bg-gradient-to-br from-smoke-400 via-smoke-800 to-charcoal-400"
      />
      <img
        v-else
        :src="asset.preview_url"
        :alt="displayName"
        class="size-full object-cover cursor-pointer"
      />

      <AssetBadgeGroup :badges="asset.badges" />
      <IconGroup
        :class="
          cn(
            'absolute top-2 right-2 invisible group-hover:visible',
            dropdownMenuButton?.isOpen && 'visible'
          )
        "
      >
        <Button
          v-tooltip.bottom="$t('assetBrowser.modelInfo.title')"
          :aria-label="$t('assetBrowser.modelInfo.title')"
          variant="secondary"
          size="sm"
          @click.stop="$emit('showInfo', asset)"
        >
          <i class="icon-[lucide--info]" />
        </Button>
        <MoreButton
          v-if="showAssetOptions"
          ref="dropdown-menu-button"
          size="sm"
        >
          <template #default>
            <Button
              variant="secondary"
              size="md"
              class="justify-start"
              @click="confirmDeletion"
            >
              <i class="icon-[lucide--trash-2]" />
              <span>{{ $t('g.delete') }}</span>
            </Button>
          </template>
        </MoreButton>
      </IconGroup>
    </div>
    <div class="max-h-32 flex flex-col gap-2 justify-between flex-auto">
      <h3
        :id="titleId"
        v-tooltip.top="{ value: displayName, showDelay: tooltipDelay }"
        :class="
          cn(
            'm-0 text-sm font-semibold line-clamp-2 wrap-anywhere',
            'text-base-foreground'
          )
        "
      >
        {{ displayName }}
      </h3>
      <p
        :id="descId"
        v-tooltip.top="{ value: asset.secondaryText, showDelay: tooltipDelay }"
        :class="
          cn(
            'm-0 text-sm line-clamp-2 [-webkit-box-orient:vertical] [-webkit-line-clamp:2] [display:-webkit-box] text-muted-foreground'
          )
        "
      >
        {{ asset.secondaryText }}
      </p>
      <div class="flex items-center justify-between gap-2 mt-auto">
        <div class="flex gap-3 text-xs text-muted-foreground">
          <span v-if="asset.stats.stars" class="flex items-center gap-1">
            <i class="icon-[lucide--star] size-3" />
            {{ asset.stats.stars }}
          </span>
          <span
            v-if="asset.stats.downloadCount"
            class="flex items-center gap-1"
          >
            <i class="icon-[lucide--download] size-3" />
            {{ asset.stats.downloadCount }}
          </span>
          <span
            v-if="asset.stats.formattedDate"
            class="flex items-center gap-1"
          >
            <i class="icon-[lucide--clock] size-3" />
            {{ asset.stats.formattedDate }}
          </span>
        </div>
        <Button
          v-if="interactive"
          variant="secondary"
          size="lg"
          class="shrink-0 relative"
          @click.stop="handleSelect"
        >
          {{ $t('g.use') }}
          <StatusBadge
            v-if="isNewlyImported"
            severity="contrast"
            class="absolute -top-0.5 -right-0.5"
          />
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useImage } from '@vueuse/core'
import { computed, ref, toValue, useId, useTemplateRef } from 'vue'
import { useI18n } from 'vue-i18n'

import IconGroup from '@/components/button/IconGroup.vue'
import MoreButton from '@/components/button/MoreButton.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import { showConfirmDialog } from '@/components/dialog/confirm/confirmDialog'
import Button from '@/components/ui/button/Button.vue'
import AssetBadgeGroup from '@/platform/assets/components/AssetBadgeGroup.vue'
import type { AssetDisplayItem } from '@/platform/assets/composables/useAssetBrowser'
import { assetService } from '@/platform/assets/services/assetService'
import { getAssetDisplayName } from '@/platform/assets/utils/assetMetadataUtils'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useAssetDownloadStore } from '@/stores/assetDownloadStore'
import { useDialogStore } from '@/stores/dialogStore'
import { cn } from '@/utils/tailwindUtil'

const { asset, interactive, focused } = defineProps<{
  asset: AssetDisplayItem
  interactive?: boolean
  focused?: boolean
}>()

const emit = defineEmits<{
  focus: [asset: AssetDisplayItem]
  select: [asset: AssetDisplayItem]
  deleted: [asset: AssetDisplayItem]
  showInfo: [asset: AssetDisplayItem]
}>()

const { t } = useI18n()
const settingStore = useSettingStore()
const { closeDialog } = useDialogStore()
const { isDownloadedThisSession, acknowledgeAsset } = useAssetDownloadStore()

const dropdownMenuButton = useTemplateRef<InstanceType<typeof MoreButton>>(
  'dropdown-menu-button'
)

const titleId = useId()
const descId = useId()

const displayName = computed(() => getAssetDisplayName(asset))

const isNewlyImported = computed(() => isDownloadedThisSession(asset.id))

const showAssetOptions = computed(() => !(asset.is_immutable ?? true))

const tooltipDelay = computed<number>(() =>
  settingStore.get('LiteGraph.Node.TooltipDelay')
)

const { isLoading, error } = useImage({
  src: asset.preview_url ?? '',
  alt: asset.name
})

function handleSelect() {
  acknowledgeAsset(asset.id)
  emit('select', asset)
}

function confirmDeletion() {
  dropdownMenuButton.value?.hide()
  const assetName = toValue(displayName)
  const promptText = ref<string>(t('assetBrowser.deletion.body'))
  const optionsDisabled = ref(false)
  const confirmDialog = showConfirmDialog({
    headerProps: {
      title: t('assetBrowser.deletion.header')
    },
    props: {
      promptText
    },
    footerProps: {
      confirmText: t('g.delete'),
      // TODO: These need to be put into the new Button Variants once we have them.
      confirmClass: cn(
        'bg-danger-200 text-base-foreground hover:bg-danger-200/80 focus:bg-danger-200/80 focus:ring ring-base-foreground'
      ),
      optionsDisabled,
      onCancel: () => {
        closeDialog(confirmDialog)
      },
      onConfirm: async () => {
        optionsDisabled.value = true
        try {
          promptText.value = t('assetBrowser.deletion.inProgress', {
            assetName
          })
          await assetService.deleteAsset(asset.id)
          promptText.value = t('assetBrowser.deletion.complete', {
            assetName
          })
          // Give a second for the completion message
          await new Promise((resolve) => setTimeout(resolve, 1_000))
          emit('deleted', asset)
        } catch (err: unknown) {
          console.error(err)
          promptText.value = t('assetBrowser.deletion.failed', {
            assetName
          })
          // Give a second for the completion message
          await new Promise((resolve) => setTimeout(resolve, 3_000))
        } finally {
          closeDialog(confirmDialog)
        }
      }
    }
  })
}
</script>
