<template>
  <Button
    v-tooltip.top="
      hasDisabledUpdatePacks ? $t('manager.disabledNodesWontUpdate') : null
    "
    variant="primary"
    :size
    :disabled="isUpdating"
    @click="updateAllPacks"
  >
    <DotSpinner v-if="isUpdating" duration="1s" :size="12" />
    <i v-else class="icon-[lucide--refresh-cw]" />
    <span>{{
      nodePacks.length > 1 ? $t('manager.updateAll') : $t('manager.update')
    }}</span>
  </Button>
</template>

<script setup lang="ts">
import { ref } from 'vue'

import DotSpinner from '@/components/common/DotSpinner.vue'
import Button from '@/components/ui/button/Button.vue'
import type { ButtonVariants } from '@/components/ui/button/button.variants'
import type { components } from '@/types/comfyRegistryTypes'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

type NodePack = components['schemas']['Node']

const {
  nodePacks,
  hasDisabledUpdatePacks,
  size = 'sm'
} = defineProps<{
  nodePacks: NodePack[]
  hasDisabledUpdatePacks?: boolean
  size?: ButtonVariants['size']
}>()

const isUpdating = ref<boolean>(false)

const managerStore = useComfyManagerStore()

const createPayload = (updateItem: NodePack) => {
  return {
    id: updateItem.id!,
    version: updateItem.latest_version!.version!
  }
}

const updatePack = async (item: NodePack) => {
  if (!item.id || !item.latest_version?.version) {
    console.warn('Pack missing required id or version:', item)
    return
  }
  await managerStore.updatePack.call(createPayload(item))
}

const updateAllPacks = async () => {
  if (!nodePacks?.length) {
    console.warn('No packs provided for update')
    return
  }
  isUpdating.value = true
  const updatablePacks = nodePacks.filter((pack) =>
    managerStore.isPackInstalled(pack.id)
  )
  if (!updatablePacks.length) {
    isUpdating.value = false
    return
  }
  try {
    await Promise.all(updatablePacks.map(updatePack))
    managerStore.updatePack.clear()
  } catch (error) {
    console.error('Pack update failed:', error)
    console.error(
      'Failed packs info:',
      updatablePacks.map((p) => p.id)
    )
  } finally {
    isUpdating.value = false
  }
}
</script>
