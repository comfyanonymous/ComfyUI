<template>
  <div class="flex h-full items-center justify-center p-8">
    <div class="max-w-screen p-2 lg:w-96">
      <!-- Header -->
      <div class="mb-8 flex flex-col gap-4">
        <h1 class="my-0 text-xl leading-normal font-medium">
          {{ t('auth.signup.title') }}
        </h1>
        <p class="my-0 text-base">
          <span class="text-muted">{{
            t('auth.signup.alreadyHaveAccount')
          }}</span>
          <span
            class="ml-1 cursor-pointer text-blue-500"
            @click="navigateToLogin"
            >{{ t('auth.signup.signIn') }}</span
          >
        </p>
      </div>

      <Message v-if="!isSecureContext" severity="warn" class="mb-4">
        {{ t('auth.login.insecureContextWarning') }}
      </Message>

      <template v-if="!showEmailForm">
        <p v-if="isFreeTierEnabled" class="mb-4 text-sm text-muted-foreground">
          {{
            freeTierCredits
              ? t('auth.login.freeTierDescription', {
                  credits: freeTierCredits
                })
              : t('auth.login.freeTierDescriptionGeneric')
          }}
        </p>

        <!-- OAuth Buttons (primary) -->
        <div class="flex flex-col gap-4">
          <div class="relative">
            <Button type="button" class="h-10 w-full" @click="signInWithGoogle">
              <i class="pi pi-google mr-2"></i>
              {{ t('auth.signup.signUpWithGoogle') }}
            </Button>
            <span
              v-if="isFreeTierEnabled"
              class="absolute -top-2.5 -right-2.5 rounded-full bg-yellow-400 px-2 py-0.5 text-[10px] font-bold whitespace-nowrap text-gray-900"
            >
              {{ t('auth.login.freeTierBadge') }}
            </span>
          </div>

          <Button
            type="button"
            class="h-10 bg-[#2d2e32]"
            variant="secondary"
            @click="signInWithGithub"
          >
            <i class="pi pi-github mr-2"></i>
            {{ t('auth.signup.signUpWithGithub') }}
          </Button>
        </div>

        <div class="mt-6 text-center">
          <Button
            variant="muted-textonly"
            class="text-sm underline"
            @click="switchToEmailForm"
          >
            {{ t('auth.login.useEmailInstead') }}
          </Button>
        </div>
      </template>

      <template v-else>
        <Message v-if="isFreeTierEnabled" severity="warn" class="mb-4">
          {{ t('auth.signup.emailNotEligibleForFreeTier') }}
        </Message>

        <Message v-if="userIsInChina" severity="warn" class="mb-4">
          {{ t('auth.signup.regionRestrictionChina') }}
        </Message>
        <SignUpForm v-else :auth-error="authError" @submit="signUpWithEmail" />

        <div class="mt-4 text-center">
          <Button
            variant="muted-textonly"
            class="text-sm underline"
            @click="switchToSocialLogin"
          >
            {{ t('auth.login.backToSocialLogin') }}
          </Button>
        </div>
      </template>

      <!-- Terms & Contact -->
      <p class="mt-5 text-sm text-gray-600">
        {{ t('auth.login.termsText') }}
        <a
          href="https://www.comfy.org/terms-of-service"
          target="_blank"
          class="cursor-pointer text-blue-400 no-underline"
        >
          {{ t('auth.login.termsLink') }}
        </a>
        {{ t('auth.login.andText') }}
        <a
          href="https://www.comfy.org/privacy-policy"
          target="_blank"
          class="cursor-pointer text-blue-400 no-underline"
        >
          {{ t('auth.login.privacyLink') }} </a
        >.
      </p>
      <p class="mt-2 text-sm text-gray-600">
        {{ t('cloudWaitlist_questionsText') }}
        <a
          href="https://support.comfy.org"
          class="cursor-pointer text-blue-400 no-underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          {{ t('cloudWaitlist_contactLink') }}</a
        >.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import Message from 'primevue/message'
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import SignUpForm from '@/components/dialog/content/signin/SignUpForm.vue'
import Button from '@/components/ui/button/Button.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useFreeTierOnboarding } from '@/platform/cloud/onboarding/composables/useFreeTierOnboarding'
import { getSafePreviousFullPath } from '@/platform/cloud/onboarding/utils/previousFullPath'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'
import { useToastStore } from '@/platform/updates/common/toastStore'
import type { SignUpData } from '@/schemas/signInSchema'
import { isInChina } from '@/utils/networkUtil'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const authActions = useFirebaseAuthActions()
const isSecureContext = globalThis.isSecureContext
const authError = ref('')
const userIsInChina = ref(false)
const toastStore = useToastStore()
const telemetry = useTelemetry()
const {
  showEmailForm,
  freeTierCredits,
  isFreeTierEnabled,
  switchToEmailForm,
  switchToSocialLogin
} = useFreeTierOnboarding()

const navigateToLogin = async () => {
  await router.push({ name: 'cloud-login', query: route.query })
}

const onSuccess = async () => {
  toastStore.add({
    severity: 'success',
    summary: 'Sign up Completed',
    life: 2000
  })

  const previousFullPath = getSafePreviousFullPath(route.query)
  if (previousFullPath) {
    await router.replace(previousFullPath)
    return
  }

  // Default redirect to the normal onboarding flow
  await router.push({ path: '/', query: route.query })
}

const signInWithGoogle = async () => {
  authError.value = ''
  if (await authActions.signInWithGoogle()) {
    await onSuccess()
  }
}

const signInWithGithub = async () => {
  authError.value = ''
  if (await authActions.signInWithGithub()) {
    await onSuccess()
  }
}

const signUpWithEmail = async (values: SignUpData) => {
  authError.value = ''
  if (await authActions.signUpWithEmail(values.email, values.password)) {
    await onSuccess()
  }
}

onMounted(async () => {
  // Track signup screen opened
  if (isCloud) {
    telemetry?.trackSignupOpened()
  }

  userIsInChina.value = await isInChina()
})
</script>
<style scoped>
:deep(.p-inputtext) {
  border: none !important;
  box-shadow: none !important;
  background: #2d2e32 !important;
}

:deep(.p-password input) {
  border: none !important;
  box-shadow: none !important;
}
:deep(.p-checkbox-checked .p-checkbox-box) {
  background-color: #f0ff41 !important;
  border-color: #f0ff41 !important;
}
</style>
