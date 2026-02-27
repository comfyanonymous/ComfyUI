<template>
  <div class="flex h-full items-center justify-center p-6">
    <div class="max-w-[100vw] text-center lg:w-[500px]">
      <h2 class="mb-3 text-xl text-text-primary">
        {{ $t('cloudOnboarding.authTimeout.title') }}
      </h2>
      <p class="mb-5 text-muted">
        {{ $t('cloudOnboarding.authTimeout.message') }}
      </p>

      <!-- Troubleshooting Section -->
      <div class="mb-4 rounded bg-secondary-background px-3 py-2 text-left">
        <h3 class="mb-2 text-sm font-semibold text-text-primary">
          {{ $t('cloudOnboarding.authTimeout.troubleshooting') }}
        </h3>
        <ul class="space-y-1.5 text-sm text-muted">
          <li
            v-for="(cause, index) in $tm('cloudOnboarding.authTimeout.causes')"
            :key="index"
            class="flex gap-2"
          >
            <span>•</span>
            <span>{{ cause }}</span>
          </li>
        </ul>
      </div>

      <!-- Technical Details (Collapsible) -->
      <div v-if="errorMessage" class="mb-4 text-left">
        <button
          class="flex w-full items-center justify-between rounded bg-secondary-background px-4 py-2 text-sm text-text-secondary transition-colors hover:bg-secondary-background-hover border-0"
          @click="showTechnicalDetails = !showTechnicalDetails"
        >
          <span>{{ $t('cloudOnboarding.authTimeout.technicalDetails') }}</span>
          <i
            :class="[
              'pi',
              showTechnicalDetails ? 'pi-chevron-up' : 'pi-chevron-down'
            ]"
          ></i>
        </button>
        <div
          v-if="showTechnicalDetails"
          class="mt-2 rounded border-muted-background border p-4 font-mono text-xs text-muted-foreground break-all"
        >
          {{ errorMessage }}
        </div>
      </div>

      <!-- Helpful Links -->
      <p class="mb-5 text-center text-sm text-gray-600">
        {{ $t('cloudOnboarding.authTimeout.helpText') }}
        <a
          href="https://support.comfy.org"
          class="cursor-pointer text-blue-400 no-underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          {{ $t('cloudOnboarding.authTimeout.supportLink') }}</a
        >.
      </p>

      <div class="flex flex-col gap-3">
        <Button class="w-full" @click="handleRestart">
          {{ $t('cloudOnboarding.authTimeout.restart') }}
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import Button from '@/components/ui/button/Button.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'

interface Props {
  errorMessage?: string
}

defineProps<Props>()

const router = useRouter()
const { logout } = useFirebaseAuthActions()
const showTechnicalDetails = ref(false)

const handleRestart = async () => {
  await logout()
  await router.replace({ name: 'cloud-login' })
}
</script>
