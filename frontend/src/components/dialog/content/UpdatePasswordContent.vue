<template>
  <Form
    class="flex w-96 flex-col gap-6"
    :resolver="zodResolver(updatePasswordSchema)"
    @submit="onSubmit"
  >
    <PasswordFields />

    <!-- Submit Button -->
    <Button type="submit" class="mt-4 h-10 font-medium" :loading="loading">
      {{ $t('userSettings.updatePassword') }}
    </Button>
  </Form>
</template>

<script setup lang="ts">
import type { FormSubmitEvent } from '@primevue/forms'
import { Form } from '@primevue/forms'
import { zodResolver } from '@primevue/forms/resolvers/zod'
import { ref } from 'vue'

import PasswordFields from '@/components/dialog/content/signin/PasswordFields.vue'
import Button from '@/components/ui/button/Button.vue'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { updatePasswordSchema } from '@/schemas/signInSchema'

const authActions = useFirebaseAuthActions()
const loading = ref(false)

const { onSuccess } = defineProps<{
  onSuccess: () => void
}>()

const onSubmit = async (event: FormSubmitEvent) => {
  if (event.valid) {
    loading.value = true
    try {
      await authActions.updatePassword(event.values.password)
      onSuccess()
    } finally {
      loading.value = false
    }
  }
}
</script>
