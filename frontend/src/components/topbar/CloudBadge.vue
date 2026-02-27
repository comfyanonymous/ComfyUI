<template>
  <TopbarBadge
    :badge="cloudBadge"
    :display-mode="displayMode"
    :reverse-order="reverseOrder"
    :no-padding="noPadding"
    :background-color="backgroundColor"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { TopbarBadge as TopbarBadgeType } from '@/types/comfy'

import TopbarBadge from './TopbarBadge.vue'

const {
  displayMode = 'full',
  reverseOrder = false,
  noPadding = false,
  backgroundColor = 'var(--comfy-menu-bg)'
} = defineProps<{
  displayMode?: 'full' | 'compact' | 'icon-only'
  reverseOrder?: boolean
  noPadding?: boolean
  backgroundColor?: string
}>()

const { t } = useI18n()

const cloudBadge = computed<TopbarBadgeType>(() => ({
  label: t('g.beta'),
  text: 'Comfy Cloud'
}))
</script>
