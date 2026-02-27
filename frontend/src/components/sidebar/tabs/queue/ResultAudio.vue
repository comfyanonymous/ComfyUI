<template>
  <audio controls width="100%" height="100%">
    <source :src="url" :type="htmlAudioType" />
    {{ $t('g.audioFailedToLoad') }}
  </audio>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import type { ResultItemImpl } from '@/stores/queueStore'

const { result } = defineProps<{
  result: ResultItemImpl
}>()

const url = computed(() => result.url)
const htmlAudioType = computed(() => result.htmlAudioType)
</script>
