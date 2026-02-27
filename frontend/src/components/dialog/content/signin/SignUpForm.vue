<template>
  <Form
    v-slot="$form"
    class="flex flex-col gap-6"
    :resolver="zodResolver(signUpSchema)"
    @submit="onSubmit"
  >
    <!-- Email Field -->
    <FormField v-slot="$field" name="email" class="flex flex-col gap-2">
      <label
        class="mb-2 text-base font-medium opacity-80"
        for="comfy-org-sign-up-email"
      >
        {{ t('auth.signup.emailLabel') }}
      </label>
      <InputText
        pt:root:id="comfy-org-sign-up-email"
        pt:root:autocomplete="email"
        class="h-10"
        type="text"
        :placeholder="t('auth.signup.emailPlaceholder')"
        :invalid="$field.invalid"
      />
      <small v-if="$field.error" class="text-red-500">{{
        $field.error.message
      }}</small>
    </FormField>

    <PasswordFields />

    <!-- Submit Button -->
    <ProgressSpinner v-if="loading" class="mx-auto h-8 w-8" />
    <Button
      v-else
      type="submit"
      class="mt-4 h-10 font-medium"
      :disabled="!$form.valid"
    >
      {{ t('auth.signup.signUpButton') }}
    </Button>
  </Form>
</template>

<script setup lang="ts">
import type { FormSubmitEvent } from '@primevue/forms'
import { Form, FormField } from '@primevue/forms'
import { zodResolver } from '@primevue/forms/resolvers/zod'
import { useThrottleFn } from '@vueuse/core'
import InputText from 'primevue/inputtext'
import ProgressSpinner from 'primevue/progressspinner'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { signUpSchema } from '@/schemas/signInSchema'
import type { SignUpData } from '@/schemas/signInSchema'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

import PasswordFields from './PasswordFields.vue'

const { t } = useI18n()
const authStore = useFirebaseAuthStore()
const loading = computed(() => authStore.loading)

const emit = defineEmits<{
  submit: [values: SignUpData]
}>()

const onSubmit = useThrottleFn((event: FormSubmitEvent) => {
  if (event.valid) {
    emit('submit', event.values as SignUpData)
  }
}, 1_500)
</script>
