<template>
  <Button
    v-tooltip.top="$t('manager.tryUpdateTooltip')"
    variant="primary"
    :size
    :disabled="isUpdating"
    @click="tryUpdate"
  >
    <DotSpinner
      v-if="isUpdating"
      duration="1s"
      :size="size === 'sm' ? 12 : 16"
    />
    <i v-else class="icon-[lucide--refresh-cw]" />
    <span>{{ isUpdating ? t('g.updating') : t('manager.tryUpdate') }}</span>
  </Button>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import DotSpinner from '@/components/common/DotSpinner.vue'
import Button from '@/components/ui/button/Button.vue'
import type { ButtonVariants } from '@/components/ui/button/button.variants'
import type { components } from '@/types/comfyRegistryTypes'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

type NodePack = components['schemas']['Node']

const { nodePack, size } = defineProps<{
  nodePack: NodePack
  size?: ButtonVariants['size']
}>()

const { t } = useI18n()
const managerStore = useComfyManagerStore()

const isUpdating = ref(false)

async function tryUpdate() {
  if (!nodePack.id) {
    console.warn('Pack missing required id:', nodePack)
    return
  }

  isUpdating.value = true
  try {
    await managerStore.updatePack.call({
      id: nodePack.id,
      version: 'nightly'
    })
    managerStore.updatePack.clear()
  } catch (error) {
    console.error('Nightly update failed:', error)
  } finally {
    isUpdating.value = false
  }
}
</script>
