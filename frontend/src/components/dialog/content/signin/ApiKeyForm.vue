<template>
  <div class="flex flex-col gap-6">
    <div class="mb-8 flex flex-col gap-4">
      <h1 class="my-0 text-2xl leading-normal font-medium">
        {{ t('auth.apiKey.title') }}
      </h1>
      <div class="flex flex-col gap-2">
        <p class="my-0 text-base text-muted">
          {{ t('auth.apiKey.description') }}
        </p>
        <a
          href="https://docs.comfy.org/interface/user#logging-in-with-an-api-key"
          target="_blank"
          class="cursor-pointer text-blue-500"
        >
          {{ t('g.learnMore') }}
        </a>
      </div>
    </div>

    <Form
      v-slot="$form"
      class="flex flex-col gap-6"
      :resolver="zodResolver(apiKeySchema)"
      @submit="onSubmit"
    >
      <Message v-if="$form.apiKey?.invalid" severity="error" class="mb-4">
        {{ $form.apiKey.error.message }}
      </Message>

      <div class="flex flex-col gap-2">
        <label
          class="mb-2 text-base font-medium opacity-80"
          for="comfy-org-api-key"
        >
          {{ t('auth.apiKey.label') }}
        </label>
        <div class="flex flex-col gap-2">
          <InputText
            pt:root:id="comfy-org-api-key"
            pt:root:autocomplete="off"
            class="h-10"
            name="apiKey"
            type="password"
            :placeholder="t('auth.apiKey.placeholder')"
            :invalid="$form.apiKey?.invalid"
          />
          <small class="text-muted">
            {{ t('auth.apiKey.helpText') }}
            <a
              :href="`${comfyPlatformBaseUrl}/login`"
              target="_blank"
              class="cursor-pointer text-blue-500"
            >
              {{ t('auth.apiKey.generateKey') }}
            </a>
            <span class="mx-1">•</span>
            <a
              href="https://docs.comfy.org/tutorials/api-nodes/overview#log-in-with-api-key-on-non-whitelisted-websites"
              target="_blank"
              class="cursor-pointer text-blue-500"
            >
              {{ t('auth.apiKey.whitelistInfo') }}
            </a>
          </small>
        </div>
      </div>

      <div class="mt-4 flex items-center justify-between">
        <Button type="button" variant="textonly" @click="$emit('back')">
          {{ t('g.back') }}
        </Button>
        <Button
          type="submit"
          variant="primary"
          :loading="loading"
          :disabled="loading"
        >
          {{ t('g.save') }}
        </Button>
      </div>
    </Form>
  </div>
</template>

<script setup lang="ts">
import type { FormSubmitEvent } from '@primevue/forms'
import { Form } from '@primevue/forms'
import { zodResolver } from '@primevue/forms/resolvers/zod'
import InputText from 'primevue/inputtext'
import Message from 'primevue/message'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { getComfyPlatformBaseUrl } from '@/config/comfyApi'
import {
  configValueOrDefault,
  remoteConfig
} from '@/platform/remoteConfig/remoteConfig'
import { apiKeySchema } from '@/schemas/signInSchema'
import { useApiKeyAuthStore } from '@/stores/apiKeyAuthStore'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

const authStore = useFirebaseAuthStore()
const apiKeyStore = useApiKeyAuthStore()
const loading = computed(() => authStore.loading)
const comfyPlatformBaseUrl = computed(() =>
  configValueOrDefault(
    remoteConfig.value,
    'comfy_platform_base_url',
    getComfyPlatformBaseUrl()
  )
)

const { t } = useI18n()

const emit = defineEmits<{
  (e: 'back'): void
  (e: 'success'): void
}>()

const onSubmit = async (event: FormSubmitEvent) => {
  if (event.valid) {
    await apiKeyStore.storeApiKey(event.values.apiKey)
    emit('success')
  }
}
</script>
