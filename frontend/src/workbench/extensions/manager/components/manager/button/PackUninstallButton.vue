<template>
  <Button variant="destructive" :size @click="uninstallItems">
    <i class="icon-[lucide--trash-2]" />
    {{
      nodePacks.length > 1
        ? t('manager.uninstallSelected')
        : t('manager.uninstall')
    }}
  </Button>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import type { ButtonVariants } from '@/components/ui/button/button.variants'
import type { components } from '@/types/comfyRegistryTypes'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import type { components as ManagerComponents } from '@/workbench/extensions/manager/types/generatedManagerTypes'

type NodePack = components['schemas']['Node']

const { nodePacks, size } = defineProps<{
  nodePacks: NodePack[]
  size?: ButtonVariants['size']
}>()

const managerStore = useComfyManagerStore()
const { t } = useI18n()

const createPayload = (
  uninstallItem: NodePack
): ManagerComponents['schemas']['ManagerPackInfo'] => {
  if (!uninstallItem.id) {
    throw new Error('Node ID is required for uninstallation')
  }

  return {
    id: uninstallItem.id,
    version: uninstallItem.latest_version?.version || 'unknown'
  }
}

const uninstallPack = (item: NodePack) =>
  managerStore.uninstallPack(createPayload(item))

const uninstallItems = async () => {
  if (!nodePacks?.length) return
  await Promise.all(nodePacks.map(uninstallPack))
}
</script>
