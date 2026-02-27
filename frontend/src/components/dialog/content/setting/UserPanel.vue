<template>
  <div class="user-settings-container h-full">
    <div class="flex h-full flex-col">
      <h2 class="mb-2 text-2xl font-bold">{{ $t('userSettings.title') }}</h2>
      <Divider class="mb-3" />

      <!-- Normal User Panel -->
      <div v-if="isLoggedIn" class="flex flex-col gap-2">
        <UserAvatar
          v-if="userPhotoUrl"
          :photo-url="userPhotoUrl"
          shape="circle"
          size="large"
        />

        <div class="flex flex-col gap-0.5">
          <h3 class="font-medium">
            {{ $t('userSettings.name') }}
          </h3>
          <div class="text-muted">
            {{ userDisplayName || $t('userSettings.notSet') }}
          </div>
        </div>

        <div class="flex flex-col gap-0.5">
          <h3 class="font-medium">
            {{ $t('userSettings.email') }}
          </h3>
          <span class="text-muted">
            {{ userEmail }}
          </span>
        </div>

        <div class="flex flex-col gap-0.5">
          <h3 class="font-medium">
            {{ $t('userSettings.provider') }}
          </h3>
          <div class="flex items-center gap-1 text-muted">
            <i :class="providerIcon" />
            {{ providerName }}
            <Button
              v-if="isEmailProvider"
              v-tooltip="{
                value: $t('userSettings.updatePassword'),
                showDelay: 300
              }"
              variant="muted-textonly"
              size="icon-sm"
              @click="dialogService.showUpdatePasswordDialog()"
            >
              <i class="pi pi-pen-to-square" />
            </Button>
          </div>
        </div>

        <ProgressSpinner
          v-if="loading"
          class="mt-4 h-8 w-8"
          style="--pc-spinner-color: #000"
        />
        <div v-else class="mt-4 flex flex-col gap-2">
          <Button class="w-32" variant="secondary" @click="handleSignOut">
            <i class="pi pi-sign-out" />
            {{ $t('auth.signOut.signOut') }}
          </Button>
          <i18n-t
            v-if="!isApiKeyLogin"
            keypath="auth.deleteAccount.contactSupport"
            tag="p"
            class="text-muted text-sm"
          >
            <template #email>
              <a href="mailto:support@comfy.org" class="underline"
                >support@comfy.org</a
              >
            </template>
          </i18n-t>
        </div>
      </div>

      <!-- Login Section -->
      <div v-else class="flex flex-col gap-4">
        <p class="text-smoke-600">
          {{ $t('auth.login.title') }}
        </p>

        <Button
          class="w-52"
          variant="primary"
          :loading="loading"
          @click="handleSignIn"
        >
          <i class="pi pi-user" />
          {{ $t('auth.login.signInOrSignUp') }}
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Divider from 'primevue/divider'
import ProgressSpinner from 'primevue/progressspinner'

import UserAvatar from '@/components/common/UserAvatar.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useDialogService } from '@/services/dialogService'

const dialogService = useDialogService()
const {
  loading,
  isLoggedIn,
  isApiKeyLogin,
  isEmailProvider,
  userDisplayName,
  userEmail,
  userPhotoUrl,
  providerName,
  providerIcon,
  handleSignOut,
  handleSignIn
} = useCurrentUser()
</script>
