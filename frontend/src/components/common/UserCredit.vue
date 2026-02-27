<template>
  <div v-if="balanceLoading" class="flex items-center gap-1">
    <div class="flex items-center gap-2">
      <Skeleton shape="circle" width="1.5rem" height="1.5rem" />
    </div>
    <div class="flex-1"></div>
    <Skeleton width="8rem" height="2rem" />
  </div>
  <div v-else class="flex items-center gap-1">
    <Tag
      v-if="!showCreditsOnly"
      severity="secondary"
      rounded
      class="p-1 text-amber-400"
    >
      <template #icon>
        <i class="icon-[lucide--component]" />
      </template>
    </Tag>
    <div :class="textClass">
      {{ showCreditsOnly ? formattedCreditsOnly : formattedBalance }}
    </div>
  </div>
</template>

<script setup lang="ts">
import Skeleton from 'primevue/skeleton'
import Tag from 'primevue/tag'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { formatCreditsFromCents } from '@/base/credits/comfyCredits'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

const { textClass, showCreditsOnly } = defineProps<{
  textClass?: string
  showCreditsOnly?: boolean
}>()

const authStore = useFirebaseAuthStore()
const balanceLoading = computed(() => authStore.isFetchingBalance)
const { t, locale } = useI18n()

const formattedBalance = computed(() => {
  const cents =
    authStore.balance?.effective_balance_micros ??
    authStore.balance?.amount_micros ??
    0
  const amount = formatCreditsFromCents({
    cents,
    locale: locale.value
  })
  return `${amount} ${t('credits.credits')}`
})

const formattedCreditsOnly = computed(() => {
  const cents =
    authStore.balance?.effective_balance_micros ??
    authStore.balance?.amount_micros ??
    0
  const amount = formatCreditsFromCents({
    cents,
    locale: locale.value,
    numberOptions: { minimumFractionDigits: 0, maximumFractionDigits: 0 }
  })
  return amount
})
</script>
