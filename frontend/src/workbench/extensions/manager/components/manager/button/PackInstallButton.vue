<template>
  <Button
    variant="primary"
    :size
    :disabled="isLoading || isInstalling"
    @click="installAllPacks"
  >
    <i
      v-if="hasConflict && !isInstalling && !isLoading"
      class="icon-[lucide--triangle-alert] text-warning-background"
    />
    <DotSpinner
      v-else-if="isLoading || isInstalling"
      duration="1s"
      :size="size === 'sm' ? 12 : 16"
    />
    <i v-else class="icon-[lucide--download]" />
    <span>{{ computedLabel }}</span>
  </Button>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import DotSpinner from '@/components/common/DotSpinner.vue'
import Button from '@/components/ui/button/Button.vue'
import type { ButtonVariants } from '@/components/ui/button/button.variants'
import type { components } from '@/types/comfyRegistryTypes'
import type { ConflictDetail } from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import { usePackInstall } from '@/workbench/extensions/manager/composables/nodePack/usePackInstall'

type NodePack = components['schemas']['Node']

const {
  nodePacks,
  isLoading = false,
  label,
  size = 'sm',
  hasConflict,
  conflictInfo
} = defineProps<{
  nodePacks: NodePack[]
  isLoading?: boolean
  label?: string
  size?: ButtonVariants['size']
  hasConflict?: boolean
  conflictInfo?: ConflictDetail[]
}>()

const { t } = useI18n()

const { isInstalling, installAllPacks } = usePackInstall(
  () => nodePacks,
  () => hasConflict,
  () => conflictInfo
)

const computedLabel = computed(() =>
  isInstalling.value
    ? t('g.installing')
    : (label ??
      (nodePacks.length > 1 ? t('manager.installSelected') : t('g.install')))
)
</script>
