<template>
  <div class="w-96 overflow-x-hidden p-2">
    <ApiKeyForm
      v-if="showApiKeyForm"
      @back="showApiKeyForm = false"
      @success="onSuccess"
    />
    <template v-else>
      <!-- Header -->
      <div class="mb-8 flex flex-col gap-4">
        <h1 class="my-0 text-2xl leading-normal font-medium">
          {{ isSignIn ? t('auth.login.title') : t('auth.signup.title') }}
        </h1>
        <p class="my-0 text-base">
          <span class="text-muted">{{
            isSignIn
              ? t('auth.login.newUser')
              : t('auth.signup.alreadyHaveAccount')
          }}</span>
          <span
            class="ml-1 cursor-pointer text-blue-500"
            @click="toggleState"
            >{{
              isSignIn ? t('auth.login.signUp') : t('auth.signup.signIn')
            }}</span
          >
        </p>
      </div>

      <Message v-if="!isSecureContext" severity="warn" class="mb-4">
        {{ t('auth.login.insecureContextWarning') }}
      </Message>

      <!-- Form -->
      <SignInForm v-if="isSignIn" @submit="signInWithEmail" />
      <template v-else>
        <Message v-if="userIsInChina" severity="warn" class="mb-4">
          {{ t('auth.signup.regionRestrictionChina') }}
        </Message>
        <SignUpForm v-else @submit="signUpWithEmail" />
      </template>

      <!-- Divider -->
      <Divider align="center" layout="horizontal" class="my-8">
        <span class="text-muted">{{ t('auth.login.orContinueWith') }}</span>
      </Divider>

      <!-- Social Login Buttons (hidden if host not whitelisted) -->
      <div class="flex flex-col gap-6">
        <template v-if="ssoAllowed">
          <Button
            type="button"
            class="h-10"
            variant="secondary"
            @click="signInWithGoogle"
          >
            <i class="pi pi-google mr-2"></i>
            {{
              isSignIn
                ? t('auth.login.loginWithGoogle')
                : t('auth.signup.signUpWithGoogle')
            }}
          </Button>

          <Button
            type="button"
            class="h-10"
            variant="secondary"
            @click="signInWithGithub"
          >
            <i class="pi pi-github mr-2"></i>
            {{
              isSignIn
                ? t('auth.login.loginWithGithub')
                : t('auth.signup.signUpWithGithub')
            }}
          </Button>
        </template>

        <Button
          type="button"
          class="h-10"
          variant="secondary"
          @click="showApiKeyForm = true"
        >
          <img
            src="/assets/images/comfy-logo-mono.svg"
            class="mr-2 h-5 w-5"
            :alt="$t('g.comfy')"
          />
          {{ t('auth.login.useApiKey') }}
        </Button>
        <small class="text-center text-muted">
          {{ t('auth.apiKey.helpText') }}
          <a
            :href="`${comfyPlatformBaseUrl}/login`"
            target="_blank"
            class="cursor-pointer text-blue-500"
          >
            {{ t('auth.apiKey.generateKey') }}
          </a>
        </small>
        <Message
          v-if="authActions.accessError.value"
          severity="info"
          icon="pi pi-info-circle"
          variant="outlined"
          closable
        >
          {{ t('toastMessages.useApiKeyTip') }}
        </Message>
      </div>

      <!-- Terms & Contact -->
      <p class="mt-8 text-xs text-muted">
        {{ t('auth.login.termsText') }}
        <a
          href="https://www.comfy.org/terms-of-service"
          target="_blank"
          class="cursor-pointer text-blue-500"
        >
          {{ t('auth.login.termsLink') }}
        </a>
        {{ t('auth.login.andText') }}
        <a
          href="https://www.comfy.org/privacy"
          target="_blank"
          class="cursor-pointer text-blue-500"
        >
          {{ t('auth.login.privacyLink') }} </a
        >.
        {{ t('auth.login.questionsContactPrefix') }}
        <a href="mailto:hello@comfy.org" class="cursor-pointer text-blue-500">
          hello@comfy.org</a
        >.
      </p>
    </template>
  </div>
</template>

<script setup lang="ts">
import Divider from 'primevue/divider'
import Message from 'primevue/message'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { getComfyPlatformBaseUrl } from '@/config/comfyApi'
import {
  configValueOrDefault,
  remoteConfig
} from '@/platform/remoteConfig/remoteConfig'
import type { SignInData, SignUpData } from '@/schemas/signInSchema'
import { isHostWhitelisted, normalizeHost } from '@/utils/hostWhitelist'
import { isInChina } from '@/utils/networkUtil'

import ApiKeyForm from './signin/ApiKeyForm.vue'
import SignInForm from './signin/SignInForm.vue'
import SignUpForm from './signin/SignUpForm.vue'

const { onSuccess } = defineProps<{
  onSuccess: () => void
}>()

const { t } = useI18n()
const authActions = useFirebaseAuthActions()
const isSecureContext = window.isSecureContext
const isSignIn = ref(true)
const showApiKeyForm = ref(false)
const ssoAllowed = isHostWhitelisted(normalizeHost(window.location.hostname))
const comfyPlatformBaseUrl = computed(() =>
  configValueOrDefault(
    remoteConfig.value,
    'comfy_platform_base_url',
    getComfyPlatformBaseUrl()
  )
)

const toggleState = () => {
  isSignIn.value = !isSignIn.value
  showApiKeyForm.value = false
}

const signInWithGoogle = async () => {
  if (await authActions.signInWithGoogle()) {
    onSuccess()
  }
}

const signInWithGithub = async () => {
  if (await authActions.signInWithGithub()) {
    onSuccess()
  }
}

const signInWithEmail = async (values: SignInData) => {
  if (await authActions.signInWithEmail(values.email, values.password)) {
    onSuccess()
  }
}

const signUpWithEmail = async (values: SignUpData) => {
  if (await authActions.signUpWithEmail(values.email, values.password)) {
    onSuccess()
  }
}

const userIsInChina = ref(false)
onMounted(async () => {
  userIsInChina.value = await isInChina()
})

onUnmounted(() => {
  authActions.accessError.value = false
})
</script>
