<template>
  <Button
    v-tooltip.top="{
      value:
        st(`commands.${normalizeI18nKey(command.id)}.label`, '') || undefined,
      showDelay: 1000
    }"
    variant="muted-textonly"
    :aria-label="st(`commands.${normalizeI18nKey(command.id)}.label`, '')"
    @click="() => commandStore.execute(command.id)"
  >
    <i
      :class="[
        typeof command.icon === 'function' ? command.icon() : command.icon
      ]"
    />
  </Button>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { st } from '@/i18n'
import type { ComfyCommand } from '@/stores/commandStore'
import { useCommandStore } from '@/stores/commandStore'
import { normalizeI18nKey } from '@/utils/formatUtil'

defineProps<{
  command: ComfyCommand
}>()

const commandStore = useCommandStore()
</script>
