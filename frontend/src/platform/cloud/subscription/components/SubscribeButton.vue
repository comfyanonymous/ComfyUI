<template>
  <Button
    :size
    :disabled="disabled"
    variant="primary"
    :style="
      variant === 'gradient'
        ? {
            background: 'var(--color-subscription-button-gradient)',
            color: 'var(--color-white)'
          }
        : undefined
    "
    :class="cn('font-bold', fluid && 'w-full')"
    @click="handleSubscribe"
  >
    {{ label || $t('subscription.required.subscribe') }}
  </Button>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { isCloud } from '@/platform/distribution/types'
import { useSubscription } from '@/platform/cloud/subscription/composables/useSubscription'
import { useTelemetry } from '@/platform/telemetry'
import { cn } from '@/utils/tailwindUtil'

const {
  size = 'lg',
  fluid = true,
  variant = 'default',
  label,
  disabled = false
} = defineProps<{
  label?: string
  size?: 'sm' | 'lg'
  variant?: 'default' | 'gradient'
  fluid?: boolean
  disabled?: boolean
}>()

const emit = defineEmits<{
  subscribed: []
}>()

const { isActiveSubscription, showSubscriptionDialog } = useBillingContext()
const { subscriptionTier } = useSubscription()
const isAwaitingStripeSubscription = ref(false)

watch(
  [isAwaitingStripeSubscription, isActiveSubscription],
  ([awaiting, isActive]) => {
    if (isCloud && awaiting && isActive) {
      emit('subscribed')
      isAwaitingStripeSubscription.value = false
    }
  }
)

const handleSubscribe = () => {
  if (isCloud) {
    useTelemetry()?.trackSubscription('subscribe_clicked', {
      current_tier: subscriptionTier.value?.toLowerCase()
    })
  }
  isAwaitingStripeSubscription.value = true
  showSubscriptionDialog()
}

onBeforeUnmount(() => {
  isAwaitingStripeSubscription.value = false
})
</script>
