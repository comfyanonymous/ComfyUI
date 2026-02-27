<template>
  <Tag :icon :severity :value />
</template>

<script setup lang="ts">
import { PrimeIcons } from '@primevue/core/api'
import Tag from 'primevue/tag'
import { computed } from 'vue'

import { t } from '@/i18n'

// Properties
const props = defineProps<{
  error: boolean
  refreshing?: boolean
}>()

// Bindings
const icon = computed(() => {
  if (props.refreshing) return PrimeIcons.QUESTION
  if (props.error) return PrimeIcons.TIMES
  return PrimeIcons.CHECK
})

const severity = computed(() => {
  if (props.refreshing) return 'info'
  if (props.error) return 'danger'
  return 'success'
})

const value = computed(() => {
  if (props.refreshing) return t('maintenance.refreshing')
  if (props.error) return t('g.error')
  return t('maintenance.OK')
})
</script>
