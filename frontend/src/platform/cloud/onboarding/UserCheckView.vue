<template>
  <CloudLoginViewSkeleton v-if="skeletonType === 'login'" />
  <CloudSurveyViewSkeleton v-else-if="skeletonType === 'survey'" />
  <CloudWaitlistViewSkeleton v-else-if="skeletonType === 'waitlist'" />
  <div v-else-if="error" class="flex h-full items-center justify-center p-8">
    <div class="max-w-[100vw] p-2 text-center lg:w-96">
      <p class="mb-4 text-red-500">{{ errorMessage }}</p>
      <Button :loading="isRetrying" class="w-full" @click="handleRetry">
        {{
          isRetrying
            ? $t('cloudOnboarding.retrying')
            : $t('cloudOnboarding.retry')
        }}
      </Button>
    </div>
  </div>
  <div v-else class="flex items-center justify-center">
    <ProgressSpinner class="h-8 w-8" />
  </div>
</template>

<script setup lang="ts">
import { useAsyncState } from '@vueuse/core'
import ProgressSpinner from 'primevue/progressspinner'
import { computed, nextTick, ref } from 'vue'
import { useRouter } from 'vue-router'

import Button from '@/components/ui/button/Button.vue'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import {
  getSurveyCompletedStatus,
  getUserCloudStatus
} from '@/platform/cloud/onboarding/auth'

import CloudLoginViewSkeleton from './skeletons/CloudLoginViewSkeleton.vue'
import CloudSurveyViewSkeleton from './skeletons/CloudSurveyViewSkeleton.vue'

const router = useRouter()
const { wrapWithErrorHandlingAsync } = useErrorHandling()
const { flags } = useFeatureFlags()
const onboardingSurveyEnabled = computed(
  () => flags.onboardingSurveyEnabled ?? true
)

const skeletonType = ref<'login' | 'survey' | 'waitlist' | 'loading'>('loading')

const {
  isLoading,
  error,
  execute: checkUserStatus
} = useAsyncState(
  wrapWithErrorHandlingAsync(async () => {
    await nextTick()

    if (!onboardingSurveyEnabled.value) {
      await router.replace({ path: '/' })
      return
    }

    const [cloudUserStats, surveyStatus] = await Promise.all([
      getUserCloudStatus(),
      getSurveyCompletedStatus()
    ])

    // Navigate based on user status
    if (!cloudUserStats) {
      skeletonType.value = 'login'
      await router.replace({ name: 'cloud-login' })
      return
    }

    // Survey is required for all users when feature flag is enabled
    if (!surveyStatus) {
      skeletonType.value = 'survey'
      await router.replace({ name: 'cloud-survey' })
      return
    }

    // User is fully onboarded (active or whitelist check disabled)
    globalThis.location.href = '/'
  }),
  null,
  { resetOnExecute: false }
)

const errorMessage = computed(() => {
  if (!error.value) return ''

  // Provide user-friendly error messages
  const errorStr = error.value.toString().toLowerCase()

  if (errorStr.includes('network') || errorStr.includes('fetch')) {
    return 'Connection problem. Please check your internet connection.'
  }

  if (errorStr.includes('timeout')) {
    return 'Request timed out. Please try again.'
  }

  return 'Unable to check account status. Please try again.'
})

const isRetrying = computed(() => isLoading.value && !!error.value)

const handleRetry = async () => {
  await checkUserStatus()
}
</script>
