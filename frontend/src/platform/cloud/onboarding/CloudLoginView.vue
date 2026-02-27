<template>
  <div class="flex h-full items-center justify-center p-8">
    <div class="max-w-screen p-2 lg:w-96">
      <!-- Header -->
      <div class="mb-8 flex flex-col gap-4">
        <h1 class="my-0 text-xl leading-normal font-medium">
          {{ t('auth.login.title') }}
        </h1>
        <i18n-t
          v-if="isFreeTierEnabled"
          keypath="auth.login.signUpFreeTierPromo"
          tag="p"
          class="my-0 text-base text-muted"
          :plural="freeTierCredits ?? undefined"
        >
          <template #signUp>
            <span
              class="cursor-pointer text-blue-500"
              @click="navigateToSignup"
              >{{ t('auth.login.signUp') }}</span
            >
          </template>
          <template #credits>{{ freeTierCredits }}</template>
        </i18n-t>
        <p v-else class="my-0 text-base text-muted">
          {{ t('auth.login.newUser') }}
          <span
            class="cursor-pointer text-blue-500"
            @click="navigateToSignup"
            >{{ t('auth.login.signUp') }}</span
          >
        </p>
      </div>

      <Message v-if="!isSecureContext" severity="warn" class="mb-4">
        {{ t('auth.login.insecureContextWarning') }}
      </Message>

      <template v-if="!showEmailForm">
        <!-- OAuth Buttons (primary) -->
        <div class="flex flex-col gap-4">
          <Button type="button" class="h-10 w-full" @click="signInWithGoogle">
            <i class="pi pi-google mr-2"></i>
            {{ t('auth.login.loginWithGoogle') }}
          </Button>

          <Button
            type="button"
            class="h-10 bg-[#2d2e32]"
            variant="secondary"
            @click="signInWithGithub"
          >
            <i class="pi pi-github mr-2"></i>
            {{ t('auth.login.loginWithGithub') }}
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
        <CloudSignInForm :auth-error="authError" @submit="signInWithEmail" />

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
    </div>
  </div>
</template>

<script setup lang="ts">
import Message from 'primevue/message'
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import Button from '@/components/ui/button/Button.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import CloudSignInForm from '@/platform/cloud/onboarding/components/CloudSignInForm.vue'
import { useFreeTierOnboarding } from '@/platform/cloud/onboarding/composables/useFreeTierOnboarding'
import { getSafePreviousFullPath } from '@/platform/cloud/onboarding/utils/previousFullPath'
import { useToastStore } from '@/platform/updates/common/toastStore'
import type { SignInData } from '@/schemas/signInSchema'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const authActions = useFirebaseAuthActions()
const isSecureContext = globalThis.isSecureContext
const authError = ref('')
const toastStore = useToastStore()
const showEmailForm = ref(false)
const { isFreeTierEnabled, freeTierCredits } = useFreeTierOnboarding()

function switchToEmailForm() {
  showEmailForm.value = true
}

function switchToSocialLogin() {
  showEmailForm.value = false
}

const navigateToSignup = async () => {
  await router.push({ name: 'cloud-signup', query: route.query })
}

const onSuccess = async () => {
  toastStore.add({
    severity: 'success',
    summary: 'Login Completed',
    life: 2000
  })

  const previousFullPath = getSafePreviousFullPath(route.query)
  if (previousFullPath) {
    await router.replace(previousFullPath)
    return
  }

  await router.push({ name: 'cloud-user-check' })
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

const signInWithEmail = async (values: SignInData) => {
  authError.value = ''
  if (await authActions.signInWithEmail(values.email, values.password)) {
    await onSuccess()
  }
}
</script>
