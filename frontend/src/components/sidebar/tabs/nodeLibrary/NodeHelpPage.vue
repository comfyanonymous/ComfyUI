<template>
  <div class="flex h-full flex-col overflow-auto">
    <div
      class="flex items-center border-b border-(--p-divider-color) px-3 py-2"
    >
      <Button
        v-tooltip.bottom="$t('g.back')"
        variant="muted-textonly"
        size="icon"
        :aria-label="$t('g.back')"
        @click="$emit('close')"
      >
        <i class="icon-[lucide--arrow-left] size-4" />
      </Button>
      <span class="ml-2 font-semibold">{{ node.display_name }}</span>
    </div>
    <div class="grow p-4">
      <NodeHelpContent :node="node" />
    </div>
  </div>
</template>

<script setup lang="ts">
import NodeHelpContent from '@/components/node/NodeHelpContent.vue'
import Button from '@/components/ui/button/Button.vue'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

const { node } = defineProps<{ node: ComfyNodeDefImpl }>()

defineEmits<{
  (e: 'close'): void
}>()
</script>
