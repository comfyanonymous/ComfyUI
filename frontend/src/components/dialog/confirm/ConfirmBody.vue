<template>
  <div
    class="flex flex-col break-words px-4 py-2 text-sm text-muted-foreground border-t border-border-default"
  >
    <p v-if="promptTextReal">
      {{ promptTextReal }}
    </p>
  </div>
</template>
<script setup lang="ts">
import { computed, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

const { promptText } = defineProps<{
  promptText?: MaybeRefOrGetter<string>
}>()

const promptTextReal = computed(() => toValue(promptText))
</script>
