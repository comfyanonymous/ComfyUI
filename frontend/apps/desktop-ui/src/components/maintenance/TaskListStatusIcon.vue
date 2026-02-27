<template>
  <ProgressSpinner v-if="!state || loading" class="h-8 w-8" />
  <template v-else>
    <i v-tooltip.top="{ value: tooltip, showDelay: 250 }" :class="cssClasses" />
  </template>
</template>

<script setup lang="ts">
import { PrimeIcons } from '@primevue/core/api'
import ProgressSpinner from 'primevue/progressspinner'
import type { MaybeRef } from 'vue'
import { computed } from 'vue'

import { t } from '@/i18n'

// Properties
const tooltip = computed(() => {
  if (props.state === 'error') {
    return t('g.error')
  } else if (props.state === 'OK') {
    return t('maintenance.OK')
  } else {
    return t('maintenance.Skipped')
  }
})

const cssClasses = computed(() => {
  let classes: string
  if (props.state === 'error') {
    classes = `${PrimeIcons.EXCLAMATION_TRIANGLE} text-red-500`
  } else if (props.state === 'OK') {
    classes = `${PrimeIcons.CHECK} text-green-500`
  } else {
    classes = PrimeIcons.MINUS
  }

  return `text-3xl pi ${classes}`
})

// Model
const props = defineProps<{
  state: 'warning' | 'error' | 'resolved' | 'OK' | 'skipped' | undefined
  loading?: MaybeRef<boolean>
}>()
</script>
