<template>
  <Button
    v-tooltip.bottom="{
      value: $t('subscription.subscribeToRunFull'),
      showDelay: 600
    }"
    class="subscribe-to-run-button whitespace-nowrap"
    variant="primary"
    size="sm"
    :style="{
      background: 'var(--color-subscription-button-gradient)',
      color: 'var(--color-white)',
      borderColor: 'transparent'
    }"
    data-testid="subscribe-to-run-button"
    @click="handleSubscribeToRun"
  >
    <i class="pi pi-lock" />
    {{ buttonLabel }}
  </Button>
</template>

<script setup lang="ts">
import { breakpointsTailwind, useBreakpoints } from '@vueuse/core'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'

const { t } = useI18n()
const breakpoints = useBreakpoints(breakpointsTailwind)
const isMdOrLarger = breakpoints.greaterOrEqual('md')

const buttonLabel = computed(() =>
  isMdOrLarger.value
    ? t('subscription.subscribeToRunFull')
    : t('subscription.subscribeToRun')
)

const { showSubscriptionDialog } = useBillingContext()

const handleSubscribeToRun = () => {
  if (isCloud) {
    useTelemetry()?.trackRunButton({ subscribe_to_run: true })
  }

  showSubscriptionDialog()
}
</script>
